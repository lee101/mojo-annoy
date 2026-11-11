"""Rerank crossover probe: serial vs `parallelize` over the candidate dists.

Builds a throwaway library that exports `probe_rerank(serial)` and calls it on
`candidate_count` x `d` float32 vectors, matching `rerank_range` exactly. Tells
us the candidate count at which a parallel rerank would beat serial on this
host, which is then compared against the candidate counts real searches reach.

Usage: `pixi run bench-rerank`
"""

from __future__ import annotations

import ctypes
import pathlib
import subprocess
import sys
import tempfile
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
SOURCE = (ROOT / "src" / "capi.mojo").read_text()

PROBE = '''
from max.algorithm import parallelize


def rerank_worker(
    vectors: F32Ptr,
    vector_norms: F32Ptr,
    query: F32Ptr,
    candidate_dist: FPtr,
    begin: Int,
    end: Int,
    d: Int,
    metric: Int,
    query_inv_norm: Float32,
):
    for c in range(begin, end):
        candidate_dist[c] = Float64(metric_rank_distance32(
            vectors + c * d,
            query,
            d,
            metric,
            vector_norms[c],
            query_inv_norm,
        ))


@export("probe_rerank")
def probe_rerank(
    vectors_addr: Int,
    vector_norms_addr: Int,
    query_addr: Int,
    candidate_dist_addr: Int,
    count: Int,
    d: Int,
    workers: Int,
) abi("C") -> Int:
    var vectors = f32ptr(vectors_addr)
    var vector_norms = f32ptr(vector_norms_addr)
    var query = f32ptr(query_addr)
    var candidate_dist = fptr(candidate_dist_addr)
    if workers <= 1:
        rerank_worker(vectors, vector_norms, query, candidate_dist, 0, count, d, 1, -1.0)
        return 1
    def part(worker: Int) {var vectors, var vector_norms, var query, var candidate_dist, var count, var d, var workers}:
        rerank_worker(
            vectors,
            vector_norms,
            query,
            candidate_dist,
            worker * count // workers,
            (worker + 1) * count // workers,
            d,
            1,
            -1.0,
        )
    parallelize(part, workers, workers)
    return 1
'''


def build_probe() -> pathlib.Path:
    directory = pathlib.Path(tempfile.mkdtemp(prefix="mann-rerank-"))
    src = directory / "probe.mojo"
    src.write_text(SOURCE.split('@export("mann_build_forest")')[0] + PROBE)
    out = directory / "probe.so"
    subprocess.run(
        ["mojo", "build", "--emit", "shared-lib", str(src), "-o", str(out)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return out


def main() -> None:
    library = ctypes.CDLL(str(build_probe()))
    library.probe_rerank.argtypes = [ctypes.c_int64] * 4 + [ctypes.c_int64] * 3
    library.probe_rerank.restype = ctypes.c_int64
    d = 32
    rng = np.random.default_rng(2026)
    normals = np.ascontiguousarray(rng.normal(size=d), dtype=np.float32)
    query = np.ascontiguousarray(rng.normal(size=d), dtype=np.float32)
    print("euclidean rerank, d = 32, seconds per call (best of 21)")
    print()
    print("| candidates | serial us | 2w | 4w | 8w |")
    print("|---:|---:|---:|---:|---:|")
    for count in (121, 500, 2_000, 4_096, 8_000, 16_000, 32_000, 64_000):
        vectors = np.ascontiguousarray(rng.normal(size=(count, d)), dtype=np.float32)
        norms = np.ones(count, dtype=np.float32)
        out = np.empty(count, dtype=np.float64)
        row = []
        for workers in (1, 2, 4, 8):
            best = float("inf")
            for _ in range(21):
                start = time.perf_counter()
                library.probe_rerank(
                    vectors.ctypes.data,
                    norms.ctypes.data,
                    query.ctypes.data,
                    out.ctypes.data,
                    count,
                    d,
                    workers,
                )
                best = min(best, time.perf_counter() - start)
            row.append(best * 1e6)
        print(
            f"| {count:,} | {row[0]:.1f} | {row[1] / row[0]:.2f}x | "
            f"{row[2] / row[0]:.2f}x | {row[3] / row[0]:.2f}x |",
            flush=True,
        )


if __name__ == "__main__":
    sys.exit(main())
