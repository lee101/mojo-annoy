# mojo-annoy

`mojo-annoy` is a standalone Mojo implementation of the compute-heavy core of
[Annoy](https://github.com/spotify/annoy): a random-projection forest for
approximate nearest-neighbor search. Its Python package is also named `annoy`,
so covered code can keep using:

```python
from annoy import AnnoyIndex
```

This is a real forest index, not a brute-force facade. Mojo builds balanced
random hyperplane trees, performs best-bin-first traversal, deduplicates leaf
candidates, and exactly reranks those candidates with vectorized distance
kernels.

## Coverage

The following `AnnoyIndex` API is covered:

- metrics: `angular`, `euclidean`, `manhattan`, `hamming`, and `dot`
- `add_item`, `build`, `unbuild`, `set_seed`, and `verbose`
- `get_nns_by_item`, `get_nns_by_vector`, `get_distance`,
  `get_item_vector`, `get_n_items`, and `get_n_trees`
- `save`, `load`, `unload`, and `on_disk_build`
- `search_k` and `include_distances`
- sparse item IDs, including Annoy's zero-filled `get_item_vector` behavior

The test suite compares directly with the pinned upstream `annoy`. Exact
high-budget searches and all public distance conventions match upstream;
default-budget approximate searches are tested for comparable recall.

There are deliberate compatibility limits:

- Saved indexes use a NumPy container and are not binary-compatible with
  upstream `.ann` files.
- `n_jobs` is accepted but construction is currently single-threaded.
- `prefault` is accepted but has no effect. Index files are loaded into memory
  instead of memory-mapped.
- `on_disk_build` saves after an in-memory build; it does not lower peak memory.
- Concurrent queries on the same index object are not supported.
- The tree layout and approximate neighbor set need not be identical to
  upstream for a finite `search_k`.

## Install

The repository pins the tested Mojo nightly and includes upstream Annoy for
parity tests:

```bash
pixi install
pixi run build
pixi run test
```

## Usage

Run this from the repository so Pixi activates `python/` on `PYTHONPATH`:

```bash
pixi run python - <<'PY'
from annoy import AnnoyIndex

index = AnnoyIndex(3, "angular")
index.add_item(0, [1.0, 0.0, 0.0])
index.add_item(1, [0.9, 0.1, 0.0])
index.add_item(2, [0.0, 1.0, 0.0])
index.build(10)

ids, distances = index.get_nns_by_vector(
    [1.0, 0.0, 0.0], 2, include_distances=True
)
print(ids)
print(distances)
PY
```

This prints item `0` first with distance `0.0`, followed by item `1`.

## Benchmarks

Measured by the final `pixi run bench` on this machine; each entry is the best
of three timed repetitions. The
construction benchmark excludes calls to `add_item`, and both implementations
use one build thread. “Upstream / Mojo” is elapsed upstream time divided by
elapsed Mojo time, so values above `1.0x` favor Mojo.

Machine: Intel(R) Xeon(R) CPU E5-2697 v4 @ 2.30GHz; Linux x86_64; single process

| case | mojo-annoy | upstream annoy | upstream / Mojo |
|---|---:|---:|---:|
| Build angular, 25k x 32, 10 trees | 110.49 ms | 253.39 ms | 2.29x |
| 1k angular queries, k=10, default search | 135.32 ms | 30.07 ms | 0.22x |
| 1k angular queries, k=10, search_k=1000 | 248.05 ms | 196.72 ms | 0.79x |
| 500 euclidean queries, k=10, default | 137.65 ms | 23.81 ms | 0.17x |

No GPU path is included.

## How it works

Python owns index state and NumPy allocations. Vectors are contiguous row-major
`float64`; tree links, permutations, leaf spans, presence bits, and scratch
buffers are contiguous `int64`. The FFI passes those buffers to one Mojo shared
library as integer addresses. Mojo reconstructs
`UnsafePointer[..., AnyOrigin[mut=True]]` values inside non-parametric
`@export` functions with the C ABI.

Construction chooses two seeded pivots per node, forms their separating
direction, computes each projection once, quickselects the median in place, and
continues until leaves contain at most 32 items. Cached inverse norms avoid
repeated angular normalization. Median splitting bounds tree depth and memory.
Search uses a min-heap of alternative branches ordered by hyperplane margin.
Items from reached leaves are deduplicated with a generation-mark array, then
native-width SIMD kernels rerank squared distances and take square roots only
for the final results. Query scratch buffers are reused. Reranking stays serial
for normal searches and uses four workers only above both 4,096 candidates and
262,144 candidate-dimensions. Increasing `search_k` explores more branches; a
high enough value gives exact search.
