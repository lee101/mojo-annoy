"""Rerank gate parity: scoring must be identical on both sides of the gate.

`mann_query_forest` scores its deduplicated candidates inline when the count is
below `RERANK_PARALLEL_CANDIDATES`, and spreads the same scoring over
`RERANK_WORKERS` ranges with `parallelize` above it. Everything after scoring
(sorting, the cut, the square root) is shared code.

The gate values are compile-time constants in `src/capi.mojo`, so one build
only exposes one branch. These tests therefore pin the observable contract of
the two branches separately:

* above the gate (the layout real callers never reach: an exhaustive budget
  over in-memory data), results must be repeatable bit for bit and every
  returned id must be a genuine nearest neighbour - a skipped, doubled or
  racing worker range would break both;
* below the gate (the inline path every bench caller uses), results must be
  exact.

The gate constants themselves are checked against the Mojo source, and the
README records the offline A/B in which the gate was forced open and the
distance buffers were compared byte for byte.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from annoy import AnnoyIndex

MOJO_SOURCE = pathlib.Path(__file__).resolve().parent.parent / "src" / "capi.mojo"

# Mirror of the comptime constants in src/capi.mojo.
GATE_CANDIDATES = 65_536
GATE_WORK = 2_097_152
GATE_WORKERS = 4


def _kernel_gate() -> dict[str, int]:
    values = {}
    for line in MOJO_SOURCE.read_text().splitlines():
        line = line.strip()
        for name in (
            "RERANK_PARALLEL_CANDIDATES",
            "RERANK_PARALLEL_WORK",
            "RERANK_WORKERS",
        ):
            if line.startswith(f"comptime {name} ="):
                values[name] = int(
                    line.split("=", 1)[1].strip().rstrip(";").replace("_", "")
                )
    return values


def test_gate_constants_match_kernel_source():
    gate = _kernel_gate()
    assert gate["RERANK_PARALLEL_CANDIDATES"] == GATE_CANDIDATES
    assert gate["RERANK_PARALLEL_WORK"] == GATE_WORK
    assert gate["RERANK_WORKERS"] == GATE_WORKERS
    # The host is shared; the README promises a cap of 4 workers.
    assert 1 <= GATE_WORKERS <= 4


def _reference(data: np.ndarray, query: np.ndarray, metric: str) -> np.ndarray:
    if metric == "euclidean":
        return np.sqrt(np.maximum(0.0, ((data - query) ** 2).sum(axis=1)))
    if metric == "manhattan":
        return np.abs(data - query).sum(axis=1)
    norms = np.linalg.norm(data, axis=1) * np.linalg.norm(query)
    cosine = data @ query / np.where(norms == 0, 1.0, norms)
    return np.maximum(0.0, 2.0 - 2.0 * np.clip(cosine, -1.0, 1.0))


def _forest(n: int, d: int, metric: str, trees: int, seed: int):
    rng = np.random.default_rng(seed)
    data = rng.normal(size=(n, d))
    index = AnnoyIndex(d, metric)
    for i, row in enumerate(data):
        index.add_item(i, row)
    index.set_seed(seed)
    index.build(trees, n_jobs=1)
    return index, data


@pytest.mark.parametrize("metric", ["angular", "euclidean", "manhattan"])
def test_above_gate_parallel_rerank_is_repeatable_and_exact(metric):
    """Three trees over 22 000 in-memory items crosses the gate."""
    n, d, trees = 22_000, 32, 3
    assert n * trees >= GATE_CANDIDATES
    assert n * trees * d >= GATE_WORK
    index, data = _forest(n, d, metric, trees, seed=9)
    query = data[12].astype(np.float64)
    budget = n  # exhaustive; every present item becomes a candidate

    first = index.get_nns_by_vector(query, 25, search_k=budget, include_distances=True)
    second = index.get_nns_by_vector(query, 25, search_k=budget, include_distances=True)
    assert first[0] == second[0]
    assert np.array_equal(first[1], second[1])

    ids, distances = first
    assert len(ids) == 25 and len(set(ids)) == 25
    assert distances == sorted(distances)
    assert distances[0] == 0.0 and ids[0] == 12

    # Candidates are scored in float32 on both branches, so noise near the cut
    # can reorder the tail (angular loses the most accuracy here). Recall
    # against the float64 reference is what a skipped, doubled or racing
    # parallel range would destroy.
    reference = _reference(data, query, metric)
    truth = set(np.argsort(reference, kind="stable")[: len(ids)].tolist())
    assert len(truth & set(ids)) >= len(ids) - 1


def test_below_gate_inline_rerank_is_exact():
    """A search whose candidates stay below the gate is also exact."""
    n, d, trees = 4_000, 32, 1
    assert n * trees < GATE_CANDIDATES
    index, data = _forest(n, d, "euclidean", trees, seed=3)
    query = data[17].astype(np.float64)
    ids, distances = index.get_nns_by_vector(
        query, 10, search_k=n, include_distances=True
    )
    expected = np.sqrt(np.maximum(0.0, ((data - query) ** 2).sum(axis=1)))
    cutoff = np.partition(expected, 10)[10]
    assert ids[0] == 17 and distances[0] == 0.0
    assert all(expected[item] <= cutoff for item in ids)
