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
| Build angular, 25k x 32, 10 trees | 88.04 ms | 270.56 ms | 3.07x |
| 1k angular queries, k=10, default search | 31.99 ms | 19.57 ms | 0.61x |
| 1k angular queries, k=10, search_k=1000 | 81.17 ms | 101.53 ms | 1.25x |
| 500 euclidean queries, k=10, default | 13.05 ms | 8.77 ms | 0.67x |

### GPU

**CPU only: memory-bound.** No GPU path is included, and none is warranted on
this port.

The gate comes from `bench/analysis.md` §2 and §6, which model flopped work
against bytes touched for every kernel measured at real caller sizes (25 000 ×
32 angular with 10 trees, 1 000 queries at k=10; 10 000 × 32 euclidean, 500
queries):

| kernel region | flops/byte (measured model) |
|---|---:|
| coarse internal-node `dot32`, default budget | 0.49 |
| fine candidate scan + rerank, default budget | 1.98 |
| deep search traversal (`search_k=1000`) | 0.53 |
| `mann_build_forest` tree projections | 0.65 |
| hamming distance | ~0.5 |

The highest ratio anywhere is 1.98, for the rerank inner loop at the smallest
benchmarked dimension (f=32, which is `2·f` flops over `2·f·4` bytes); it falls
toward 0.25 as f grows. That is below the roughly 2-flop-per-byte bar, so no
kernel here can absorb a transfer. The default query moves only 10–27 KB of
`normals` and `perm` and the rerank touches ~4.3 µs of a ~17–19 µs kernel, so
even a zero-cost launch would lose to the one-copy-in/copy-out cost. The
analysis attributes the hot rows to pointer-chasing latency at 1.0–3.7 GB/s
effective against a 7.8–9.4 GB/s streaming baseline, not to arithmetic.

Device check: `nvidia-smi --query-gpu=memory.free` reported 5486 MiB free when
this decision was recorded, so the shared GPU met the 4000 MiB floor; the gate
above, not device availability, is the reason for stopping. There is no device
code to measure, so there are no GPU before/after numbers: the numbers above are
the existing CPU measurements from `pixi run bench` and `bench/analysis.md`, and
both rows where the port trails upstream (angular 0.61x, euclidean 0.67x) are
attributed there to Python/ctypes glue and issue-bound traversal. No GPU tests
were written, so a test run neither exercises nor skips a GPU device.

## How it works

Python owns index state and NumPy allocations. Durable vectors and forest state use
contiguous row-major `float64`; compact search mirrors use `float32`; tree links,
permutations, leaf spans, presence bits, and scratch buffers use contiguous `int64`.
Float32 NumPy queries cross the FFI zero-copy, while other real dtypes reuse one
conversion buffer. A reusable packed argument block reduces each query to one C ABI
argument. Mojo reconstructs
`UnsafePointer[..., AnyOrigin[mut=True]]` values inside non-parametric
`@export` functions with the C ABI.

Construction chooses two seeded pivots per node, forms their separating
direction, computes each projection once, quickselects the median in place, and
continues until leaves contain at most 32 items. Cached inverse norms avoid
repeated angular normalization. Median splitting bounds tree depth and memory.
Search uses a max-heap ordered by the minimum signed normalized hyperplane margin
along each path, matching upstream Annoy's traversal invariant. Items from reached
leaves are deduplicated with a generation-mark array, then unrolled native-width
SIMD kernels rerank squared distances with scalar tails and take square roots only
for final results. Query scratch buffers are reused. Reranking stays serial for
normal searches and uses four workers only above both 4,096 candidates and 262,144
candidate-dimensions. Increasing `search_k` explores more branches; a high enough
value gives exact search.
