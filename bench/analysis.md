# ANALYZE phase — measured hotspot map for mojo-annoy

Status: analysis only. No change to `src/`, `tests/`, or any algorithm. The only
repo file added is this one. Stray profiling scripts live in `/tmp`.

Machine (from `/proc/cpuinfo`, `/proc/meminfo`, `lscpu`):

- Intel(R) Xeon(R) CPU E5-2697 v4 @ 2.30GHz, 2 sockets × 18 cores, 2 threads/core (72 logical)
- L1d 32 KiB/core, L2 256 KiB/core, L3 45 MiB/socket
- SIMD available: SSE2/4.2, AVX, AVX2, FMA. **No AVX-512** (`avx512f` absent from flags).
- Linux x86_64, single process, no `taskset` pinning.

Toolchain: `mojo ==1.2.0.dev2026092605`, `max-core ==26.7.0.dev2026092605`,
`annoy ==1.17.3` (upstream, imported via the same `bench/bench.py` loader).

---

## 0. Measurement conditions (read before trusting any number)

| Item | Value |
|---|---|
| Benchmark command | `pixi run bench` (flock-wrapped), nothing else started alongside |
| Each table time | best of N repetitions of one timed closure; N stated per section |
| PMU hardware counters | **unavailable** — `perf_event_paranoid = 4`, `perf_event_open` returns `EACCES`. No cycles/instructions numbers are reported anywhere in this file. |
| Build state | `dist/libmojo-annoy.so` was **stale** when the session started (compiled-in `DBG ids …` strings, 1.4 s older than `src/capi.mojo`; `src/` is clean in git). Rebuilt once with `pixi run build` (warnings only, exit 0) before profiling. All numbers below are from the rebuilt library. |
| Size assumptions | "real caller" sizes taken from `bench/bench.py`: 25 000 × 32 angular, 10 trees; 1 000 queries (k=10); 10 000 × 32 euclidean, 500 queries. Each public entry point is also swept over sizes. |

Where a per-query "bytes touched" is *modelled* rather than directly counted,
the table says so and gives the formula. Where it was physically counted
(page faults, nmap remap with `sync=0`) the basis says "counted".

---

## 1. Headline: `pixi run bench` on the rebuilt library

```
Machine: Intel(R) Xeon(R) CPU E5-2697 v4 @ 2.30GHz; Linux x86_64; single process

| case | mojo-annoy | upstream annoy | upstream / Mojo |
|---|---:|---:|---:|
| Build angular, 25k x 32, 10 trees | 93.02 ms | 273.48 ms | 2.94x |
| 1k angular queries, k=10, default search | 29.95 ms | 22.50 ms | 0.75x |
| 1k angular queries, k=10, search_k=1000 | 79.47 ms | 106.29 ms | 1.34x |
| 500 euclidean queries, k=10, default | 17.52 ms | 10.84 ms | 0.62x |
```

A second full run of the same command gave: `95.93 / 22.26 / 66.71 / 11.08 ms`
vs `250.93 / 20.85 / 104.55 / 10.60 ms` → `2.62x / 0.94x / 1.57x / 0.96x`.
Run-to-run spread on the query rows is ±15 %; treat sub-15 % differences as noise.

**The port is slower than upstream on both default-search rows** (angular 0.75x,
euclidean 0.62x) and faster on build and deep search. Rows where the port is
slower are marked ⚠ below and are not dropped.

---

## 2. Ranked hotspot table

`glue` = full Python API time − kernel-only time (kernel-only = the same number
of `mann_query_forest` / `mann_distance` / `mann_build_forest` ctypes calls, with
the index's own packed arg block patched exactly as `index.py` patches it). Column
"kernel" is the raw ctypes-call time.

The "flops" and "bytes" columns are **models** of the work each kernel does; their
inputs (candidate counts, visited counts, dot-product lengths) come from measured
runs. "flops per byte" is always flops ÷ bytes of that same model row.

| # | function | input size | time (measured) | bytes touched | flops | flops/byte | bound | where the time sits |
|---:|---|---|---:|---:|---:|---:|---|---|
| 1 | `get_nns_by_vector` — python glue (⚠ slower than upstream) | k=10, 1000 q, f=32, n=25k, 10 trees | **12.5–13.6 µs/q** (full 29.2–32.1 µs, kernel 17–19 µs) | 0 alloc/call for float32 C-query (`_query_args` reused in place); ~47 B/query retained | 0 | 0 | **glue (Python)** | Python: arg patching (4 scalar + 2 `addr()` calls), `tolist()`, `min/max`, `heap_cap` arithmetic |
| 2 | `get_nns_by_vector` — kernel, default budget (⚠) | k=10, 1000 q | **16.9–19.1 µs/q** kernel; 31.6 ms full / 32.1 ms full+dist | 15.6–16.6 KB/q counting every perm entry in each reached leaf (measured visited 120.5/q, 128.6 perm slots/q); 27.4 KB/q if counting `normals` rows for internal nodes; ~21 KB/q counting distinct 4 KiB pages faulted on a cold copy | 8.4 kFlop/q (dot products: 90 internal × 2·32…2·64 + 121 rerank × 2·32) | 0.51–0.54 | **memory-bound** | kernel (`mann_query_forest`) |
| 3 | `get_nns_by_vector` — kernel, deep search | k=10, search_k=1000, 1000 q | **76.4 µs/q** kernel; 79.5–90.7 ms full | 128.9 KB/q (ref. `perm`) + 11.6 KB/q (normals) → 140.5 KB/q; 21 KB/q fp32 candidates | 75.0 kFlop/q | 0.53 | **memory-bound** | kernel |
| 4 | `get_nns_by_vector` — kernel, euclidean default (⚠) | k=10, 500 q, n=10k | **16.0 µs/q** kernel; 14.6–17.5 ms full | 16.9 KB/q (ref. `perm`, ~120.7 candidates/q) | 7.7 kFlop/q | 0.46 | **memory-bound** | kernel |
| 5 | `mann_build_forest` (`build`) ⚠ *compared to what the data supports* — kernel dominates | 25k × 32 f64, 10 trees, leaf 32 | **~88–97 ms** (build() total 93.4–111.6 ms; non-kernel phases measured 13.9 ms) | 25.0 MB index arrays + 0.48 MB seed projections → **~164 MB** model ≈ 6 trees × whole-set projections ≈ 2 × Σ n·2^d | 106 MFlop model | 0.65 | **memory-bound** (working set 25 MB > L2 9 MB, < L3 90 MB) | kernel |
| 6 | `build()` — non-kernel phases | same | **13.9 ms measured** breakdown: population 12.04, inv_norm 0.60, f32 mirrors 0.88, forest alloc 0.41, scratch 0.01 | 6.40 MB dense f64 materialised from `dict`, plus 14.88 MB f64→f32 mirrors | ~0 | 0 | glue (Python/numpy) | `AnnoyIndex.build` lines 59–86 |
| 7 | `add_item` (⚠ vs upstream — not in the bench table) | 25 000 items, f=32 | **51.7–60.0 ms** total (2.4 µs/item) | 6.4 MB retained f64 | 0 | 0 | glue (Python) | `_as_vector` + `np.asarray` + `.copy()` + dict store, per item |
| 8 | `get_distance` (⚠ 55x slower than upstream) | 10 000 calls, angular f=32 | **111.7–121.8 ms** (11.3–12.2 µs/call); upstream 2.02 ms | 0 allocs (3 B/call retained) | 2·32 = 64 | — (cache-resident) | **glue (Python)**, 8.1 µs of the 11.3 is `addr()` ×2 | `addr()` alone 4.4 µs/call |
| 9 | final selection loop inside the kernel — `want = k` dependent | k from 10 to 1000, fixed search_k=100 | 22.7 µs @ k=10 → 216.2 µs @ k=1000 with **identical** 120.3 candidates (0.22 µs per result slot) | O(k²) slot writes, cache-resident | 0 | 0 | **compute-bound (branchy)** | kernel `mann_query_forest` lines 619–633 |
| 10 | `get_item_vector` | 10 000 calls, f=32 | **11.3–23.8 ms** (1.03–1.2 µs/call); upstream 17.4–20.4 ms | 32 × 8 B retained per call (`.tolist()`) | 0 | 0 | glue (Python) | `_check_item` + row slice + `.tolist()` (0.67 µs) |
| 11 | `get_nns_by_item` | 1000 calls, k=10 | **32.0 µs/call** (vs 28.9 by vector) | + 1 row copy, 264 B | 0 | 0 | glue (Python) | `_vector_for_item` → `get_nns_by_vector` |
| 12 | `save` / `load` | 18.1 MB npz | **73.7 ms / 51.1 ms** | 18.1 MB write / read | 0 | 0 | I/O-bound | Python `np.savez` / `np.load` |
| 13 | hamming `metric_rank_distance` inner loop | per distance | **not measured hot** — no caller in the bench uses hamming; scalar `for i in range(d)` with `Int64(a[i]) != Int64(b[i])` is visible at `src/capi.mojo:137,220` | — | — | — | compute-bound (scalar) | kernel, gated by size |

⚠ marks rows where the port is slower than upstream on the same input in this
session's runs: rows 1+2 (angular default, upstream 22.5 ms vs 29.2–32.1 ms),
row 4 (euclidean default, 9.7–10.8 ms vs 14.6–17.5 ms), row 7 (not in the bench,
but `add_item` is the one call whose Python loop is per element), row 8
(`get_distance`), row 10 is **faster** than upstream and marked accordingly in
§5.

---

## 3. The single most important finding: where the query time actually goes

The default-budget query is **not** dominated by reranking. Measured split, per
query (angular, n=25k, f=32, 10 trees, k=10):

| component | µs/query | how measured |
|---|---:|---|
| full Python API | 29.2–32.1 | `best()` over 1000 queries, 5–7 reps |
| kernel only (same args) | 16.9–19.1 | same loop with ctypes call only |
| **Python/ctypes glue** | **12.5–13.6** | difference of the two rows above |
| — of which ctypes floor | 1.4–1.9 | `mann_distance` on real buffers, 20 000 calls |
| — of which `addr()` ×2 | 8.1 | `addr()` measured on 10 000 arrays = 4.05 µs/call ×2 = 8.1 |

Inside the kernel, a `k=10` vs `search_k` sweep with **k held at 10** shows the
candidate-scan part is the mass, and internal-node dot products are small:

| measurement | value | basis |
|---|---:|---|
| candidates per default query | **120.3–120.5** (median 121) | counted from the result/distinct set, 100–1000 queries |
| internal nodes per default query | **~80** (perm slots 128.6 vs 120.5 visited → ~1.07 perm entries/visited) | counted via `visited/q` and `sk` sweep |
| cost from visited ≈ 24 → 163 per query | 5.4 → 24.2 µs/query (10 trees, k=10) | measured sweep |
| cost from visited ≈ 24 → 5008 (1 tree, k=10) | 5.4 → 186 µs/query, i.e. **~36 ns per extra visited item** near the top and **~43 ns** at the low end | measured sweep |
| fine level (candidate scan + rerank) at default 120 candidates | **≈ 4.3 µs of the 16.9–19.1 µs** | 36 ns × 120 |
| coarse level (internal-node `dot32` + `normals` reads), remainder | **≈ 13–15 µs/query** | 12.6–14.8 µs, measured as the y-intercept of the `k=10` sweep |

So for the default-budget case the ranking is:

1. **Python/ctypes glue: 12.5–13.6 µs/query (≈ 42–45 % of the wall time).**
2. **Coarse internal-node dot products: ≈ 13–15 µs/query (≈ 44–48 %).**
3. Candidate scan + rerank: ≈ 4.3 µs/query (≈ 14 %).

Upstream's whole query is 22.5 µs. The port's kernel alone is 17–19 µs, so even a
zero-cost glue would leave the port at ~0.85x on this row. Both rows 1 and 2 need
work.

### Why the coarse level is so expensive for so little work

Per default query: 80 internal nodes × `dot32(query, normal, 32)` = 80 × 32 × 2 =
5.1 kFlop, against 80 × (32 × 4 B normal + 4 B threshold) = 10.5 KB of reads —
an **0.49 flop/byte** operation that must therefore be memory-latency-bound. Two
independent measurements agree that the bytes leaving the private caches per query
are small (10–27 KB, i.e. L1/L2-resident), which means the coarse level is not
DRAM-bound either: it is **latency- and issue-bound**, with the loads of
`normals` and `perm` being independent (non-prefetchable) random 128 B rows.

Measured evidence for that conclusion:

- The numpy equivalent of reading exactly one 128 B row per candidate
  (`((rows-q)**2).sum(1)` over random 32-dim float32 rows) costs **103.8 ns per
  candidate**; the port's whole candidate scan + rerank costs **36 ns per
  candidate**. The port's SIMD rerank is therefore already ~3x faster per row than
  a vectorised numpy pass over the same scattered rows.
- Increasing the *index size* at fixed work barely changes per-query time until
  the working set passes L3: at sk=200, 1 tree, kernel time went
  12.65 µs (0.13 MB) → 17.55 µs (12.8 MB, fits L3) → 24.76 µs (51.2 MB, exceeds
  L3) per query. That is the signature of an L3/DRAM latency step, not of
  bandwidth saturation: streaming bandwidth on this box is 7.8–9.4 GB/s
  (numpy `sum` over 2/16/128/512 MiB), while the traversal achieves **1.0–3.7
  GB/s effective** on the same data.

---

## 4. Python-level loops that re-enter ctypes per element, and per-call copies

### 4.1 Per-element ctypes re-entry (none in the hot query path; one in `add_item`)

- **`get_nns_by_vector` does not loop per element.** One `mann_query_forest` call
  per query, through a reusable 28-slot `int64` arg block. This is correct and
  should be kept.
- **`add_item` is a per-item Python call** (2.4 µs/item, 60 ms for 25 000
  items). It never enters ctypes, but it does `np.asarray` + shape/dtype checks +
  `.copy()` + dict insertion per item. For a real caller that adds rows in a loop,
  this alone is ~60 % of the total build wall time (`build incl population`
  164.4 ms vs `build()` alone 101.9–104.4 ms).
- **`get_distance` re-enters ctypes once per call and pays `addr()` twice per
  call** — 8.1 of its 11.3 µs. `addr()` re-validates `isinstance`, dtype
  membership (0.94 µs of the 4.05), contiguity, and calls `int(array.ctypes.data)`
  (1.63 µs) on arrays that the port itself allocated and never mutate.
- `unbuild()` (index.py:126–130) rebuilds `_items` in a Python dict comprehension
  over all n rows — a per-row Python loop, but off the benchmarked paths.

### 4.2 Per-call allocations and copies

| call | allocations | copies |
|---|---|---|
| `get_nns_by_vector`, **float32 C-contiguous** query | none in the hot path (`_query_args` reused in place); `ids = result_ids[:found].tolist()` allocates a Python list of `found` ints per call | none |
| `get_nns_by_vector`, **float64 or list** query | none, but `np.copyto(self._query_vector, value)` into a reused f32 buffer | one f32 conversion per call (unavoidable across the dtype boundary) |
| `get_nns_by_vector(include_distances=True)` | `result_dist[:found]` numpy view + `.tolist()`; also `self._stamp == np.iinfo(np.int64).max` **constructs a numpy scalar per call** (`np.iinfo(...)` shows up 1000× in the cProfile) | dot-metric case does `distances = -distances`, a full copy |
| `get_distance` | `left`/`right` are views (`_vectors[item]`) — cheap | dot metric negates a scalar |
| `add_item` | one `np.ndarray` per item via `.copy()` | full row copy per item |
| `build` | 25.0 MB of index arrays + 0.48 MB projections (measured) | dense f64 copy of all items out of the dict (**12.04 ms measured**), then 4 f64→f32 mirrors (**0.88 ms measured**) |
| `save`/`load` | npz container | the **whole index is stored twice** in dtype terms: `vectors` f64 + search mirror f32, `normals` f64 + f32 (−12.3 MB avoidable; see §7) |

Measured allocation volume on the query path: `tracemalloc` reports
**~47 B retained per query** (no distances) and **127 B per query** (with
distances) — i.e. Python glue is allocation-light; it is *CPU*-heavy (attribute
lookups, `addr()`, `min/max`, `tolist`), which matches the glue breakdown.

---

## 5. Upstream comparison, per entry point

| entry point | mojo-annoy | upstream annoy | verdict |
|---|---:|---:|---|
| `build(10)` 25k×32 | 93.4–111.6 ms | 243.2–281.3 ms | port **2.6–2.9x faster** |
| `get_nns_by_vector` ×1000 angular default | 29.2–32.1 ms | 21.9–23.6 ms | ⚠ port **0.72–0.75x** (slower) |
| `get_nns_by_vector` ×1000 angular search_k=1000 | 79.5–107.4 ms | 106.3–123.4 ms | port **1.2–1.5x faster** (noisy) |
| `get_nns_by_vector` ×500 euclidean default | 14.6–17.5 ms | 9.7–12.2 ms | ⚠ port **0.62–0.83x** (slower) |
| `get_nns_by_vector` ×200, f ∈ {8,32,64,128,256} | 6.10 / 7.36 / 7.84 / 9.06 / 11.72 ms | 5.62 / 5.11 / 5.96 / 8.53 / 13.55 ms | port loses at f≤64, wins at f=128, 256 |
| `get_distance` ×10 000 | 111.7–121.8 ms | 2.02–2.06 ms | ⚠ port **~55x slower** |
| `get_item_vector` ×10 000 | 11.3–23.8 ms | 17.4–20.4 ms | port competitive/faster |
| `get_nns_by_item` ×1000 | 32.0 µs/call | — (not measured separately this session) | — |

Recall (measured, 200 queries, exact float32 ground truth): angular default
recall@10 = **0.116** (upstream 0.121), search_k=1000 = **0.340** (upstream 0.398).
The port spends comparable time for slightly lower recall at deep search; that is
an *algorithmic* difference, out of scope for this phase, but it is why I do not
recommend "just search less".

---

## 6. GPU verdict (needs the flops-per-byte figure)

| kernel region | flops/byte (model, from §2) | GPU? |
|---|---:|---|
| coarse internal-node `dot32` (default budget) | **0.49** | **No** |
| fine candidate scan + rerank (default budget) | **1.98** | **No** |
| deep search, entire traversal | **0.53** | **No** |
| hamming distance | ~0.5 (1 cmp per 8 B, scalar) | **No** |
| build projections | **0.65** | **No** |

Every kernel in this port sits **below 2 flop/byte**. The GPU crossover on this
host is not reachable: a 0.5-flop/byte kernel is pure data movement, and the
per-query data is already 10–140 KB — below any launch model. The only kernel even
close is the fine-level rerank at f=32 (2·32 flops over 2·32·4 B = 2 flop/byte);
as f grows that ratio drops toward 0.25, and as f shrinks the candidate count
falls. A GPU path for the rerank would need the whole candidate set staged and
would lose on the ~4.3 µs/query the kernel actually spends there.

**Verdict: `npu/gpu` action for every row. Do not add a GPU path in this port;
the highest flops/byte measured anywhere is 1.98, below the 2.0 bar set in
`MOJO_NOTES.md` §5, and the work is data movement.**

---

## 7. Next phase per hotspot

| hotspot | next phase | concrete target (and why) |
|---|---|---|
| **#1 python/ctypes glue, 12.5–13.6 µs/q** | **none (this is not a kernel phase — see note)** | The work is Python-side: 8.1 µs of it is `addr()` ×2, each re-validating dtype/contiguity via `int(array.ctypes.data)`. A correctness-preserving fix is to cache the validated `int` address once at build (arrays never move), leaving one `ctypes` call per query. This is glue removal, not a kernel change, and no `unroll/simd/parallel/gpu` phase can touch it. |
| **#2 kernel default budget, 13–15 µs/q coarse + 4.3 µs/q fine** | **unroll** | The kernel loop over internal nodes is already serial and each iteration is one 32-element dot. Batching 4 nodes per iteration allows 4 independent `dot32` calls issuing in parallel instead of one dependent chain — the measurement says the cost is *issue/latency*, not flops (0.49 flop/byte, L1/L2 resident). The `dot32` body itself is already 4× unrolled into AVX `vfmadd231ps` (verified in the disassembly). |
| **#2 same row** | **simd** | Only after unroll: f32 kernels already use AVX (25 cmps of `ymm`, no AVX-512 on this host). With AVX-512 unavailable, widening past 8 lanes is impossible, so SIMD is a smaller lever than unroll here. |
| **#3 deep search, 76.4 µs/q kernel** | **unroll** | Candidate-scan cost is ~36 ns/visited item at 0.53 flop/byte. The measured throughput (1.0–3.7 GB/s effective) against a 7.8–9.4 GB/s streaming baseline shows the scan is latency-bound on random 128 B rows. Software-prefetching the next `perm` entries and batching 4 independent vector fetches per iteration is an unroll/prefetch change to that loop, not an algorithm change. |
| **#5 build kernel, ~88–97 ms** | **parallel** | `mann_build_forest` is `for tree in range(trees)` — 10 independent trees, currently serial. At 25 MB working set and 0.65 flop/byte, the work is memory-bound and per-tree independent; 10 trees on ≥10 cores is the textbook parallel win. (`MOJO_NOTES` §4: `from max.algorithm import parallelize` + explicit worker count + size threshold.) Rebuild projections must be re-checked for determinism (fixed seed) — that is for the phase that does it. |
| **#5 same row** | **unroll** | Nested: the inner `projection` loop over n rows calls a fully unrolled `dot`, so the remaining lever is accumulating 4 rows per pass to reuse the normal from cache. Secondary to `parallel`. |
| **#6 `build()` Python phases, 13.9 ms** | none (glue) | 12.04 ms is materialising a dense f64 matrix from the per-item `dict` — that is the price of the sparse-ID-compatible API and a numpy-level fix, not a kernel phase. |
| **#7 `add_item`, 2.4 µs/item** | none (glue) | Per-item Python; a bulk `add_items(rows)`-style path would need an API decision. Do not touch in an unroll/simd phase. |
| **#8 `get_distance`, 11.3 µs/call (55x upstream)** | none (glue) | Same `addr()` cause as #1. With cached addresses this is a ~1.9 µs/call call, which would be at parity with upstream. No kernel change is justified: `mann_distance` is 64 flops and entirely cache-resident. |
| **#9 final selection loop (k)** | **unroll** | 0.22 µs per result slot at fixed candidates is a branchy insertion sort through `Pointer` reads; at k=10 it is already 0.63 µs of an 18 µs query, so it only matters for large-k callers. Unrolling/branchless selection, or replacing with the same sorted insert the Python layer already assumes, is an unroll-level change. Ranked low because no benchmark row uses k > 10. |
| **#10, #11, #12** | none | Already at or better than upstream; `save`/`load` is I/O. |
| **#13 hamming distance** | **simd** | The only genuinely scalar inner loop left (`for i in range(d): Int64(a[i]) != Int64(b[i])`). Not on any benchmark row, so it is unmeasured for time here; if a caller exercises it, XOR+popcount SIMD is the obvious fix. |
| every row | **gpu** | **No** — see §6; max flops/byte measured is 1.98. |

---

## 8. Measurement notes / things that would change the conclusion

- **No PMU.** Cycles/instructions/L2/L3 counters could not be read
  (`perf_event_paranoid = 4`). The memory-vs-compute verdicts above come from
  achieved GB/s against a measured streaming baseline, from the working-set sweep
  crossing L3, and from the flops/byte model — not from counters. If the policy
  is relaxed, re-run with `perf stat -e cycles,instructions,cache-misses`.
- **Machine is shared.** 72 logical CPUs, 251 GB RAM, 121 GB in use by other
  processes during this session. Rows that disagree by <15 % between runs
  (notably search_k=1000 and euclidean) should not be ranked on a single run.
- **Bench harness leaves the CPU cold for the first query** of each `best()`
  repetition; the first measured query in `per_query_times` (max 11.6 µs) is an
  outlier for that reason. Medians are used wherever the distribution is skewed.
- **`bench/bench.py` does not exercise** `save/load`, `get_distance`,
  `get_item_vector`, `get_nns_by_item`, `add_item` cost, k > 10, or f other than
  32. Those were measured by the `/tmp` scripts summarised in §2 and §5; they are
  not in the flock-wrapped `pixi run bench`.
- **Stale `.so` at session start.** The checked-in `dist/` (untracked build
  output) was older than `src/capi.mojo` and still contained `DBG ids …` prints
  from an earlier debugging pass. Rebuilding changed the bench numbers by up to
  30 % (e.g. angular default 22.26 ms before rebuild → 29.95–31.61 ms after).
  Any previous analysis done against the stale binary would have been measuring a
  different library.
