"""Crossover and thread-scaling measurement for the parallel build projection pass.

Not part of the test suite. It builds its own throwaway copies of the Mojo
library so it can time:
  * the shipped build (gate on) against a copy with the gate forced off
    (PROJECTION_MIN_ITEMS bumped out of reach) -> the crossover,
  * the shipped worker count against copies compiled with other worker counts
    -> whether more than the cap helps.

Each row times only `mann_build_forest`: the population is re-applied untimed
and `build()` is timed, matching `time_build` in bench.py.

Usage: `pixi run bench-parallel`
"""

from __future__ import annotations

import ctypes
import os
import pathlib
import sys
import tempfile
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

from annoy import AnnoyIndex  # noqa: E402
from annoy import _lib as _bindings  # noqa: E402

SOURCE = (ROOT / "src" / "capi.mojo").read_text()


def build_variant(min_items: int, workers: int, label: str = "") -> pathlib.Path:
    text = SOURCE.replace("comptime PROJECTION_MIN_ITEMS = MIN_ITEMS_PLACEHOLDER", f"comptime PROJECTION_MIN_ITEMS = {min_items}")
    text = text.replace("comptime PROJECTION_WORKERS = 4", f"comptime PROJECTION_WORKERS = {workers}")
    out = pathlib.Path(tempfile.mkdtemp(prefix=f"mann-n{min_items}-w{workers}-")) / "lib.so"
    src = out.with_suffix(".mojo")
    src.write_text(text)
    import subprocess

    subprocess.run(
        ["mojo", "build", "--emit", "shared-lib", str(src), "-o", str(out)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return out


def load(path: pathlib.Path) -> ctypes.CDLL:
    library = ctypes.CDLL(str(path))
    for name, (argtypes, restype) in _bindings._SIGNATURES.items():
        function = getattr(library, name)
        function.argtypes = argtypes
        function.restype = restype
    return library


def timed(index: AnnoyIndex, repeat: int) -> float:
    best = float("inf")
    for _ in range(repeat):
        index.unbuild()
        for item, row in index._items.items():
            index.add_item(item, row)
        start = time.perf_counter()
        index.build(index._trees_for_sweep, n_jobs=1)
        best = min(best, time.perf_counter() - start)
    index.unbuild()
    return best


def make(n: int, d: int, trees: int, seed: int) -> AnnoyIndex:
    rng = np.random.default_rng(seed)
    index = AnnoyIndex(d, "euclidean")
    for i, row in enumerate(rng.normal(size=(n, d))):
        index.add_item(i, row)
    index._trees_for_sweep = trees
    index.build(trees, n_jobs=1)
    return index


def main() -> None:
    sizes = [int(s) for s in os.environ.get("MANN_SWEEP_SIZES", "1000,2000,4000,8000,12000,16000,25000,50000").split(",")]
    repeat = int(os.environ.get("MANN_SWEEP_REPEAT", "7"))
    dims = int(os.environ.get("MANN_SWEEP_DIM", "32"))
    trees = int(os.environ.get("MANN_SWEEP_TREES", "10"))
    rounds = int(os.environ.get("MANN_SWEEP_ROUNDS", "3"))
    variant_workers = [int(w) for w in os.environ.get("MANN_SWEEP_WORKERS", "1,2,4,8").split(",")]
    min_items = int(os.environ.get("MANN_SWEEP_MIN_ITEMS", "1"))
    print("compiling variants ...", flush=True)
    serial = load(build_variant(10**12, 4))
    variants = {w: load(build_variant(min_items, w)) for w in variant_workers}
    baseline = serial

    print(
        f"repeat = {repeat}; rounds = {rounds}; d = {dims}; {trees} trees; "
        f"gate min items = {min_items}; only mann_build_forest timed"
    )
    print()
    header = "| n |"
    for workers in variant_workers:
        header += f" {workers}w / serial |"
    header += " best / serial |"
    print(header)
    print("|---:|" + "---:|" * (len(variant_workers) + 1))
    for n in sizes:
        index = make(n, dims, trees, 2026)
        bests = {"serial": []}
        for workers in variant_workers:
            bests[workers] = []
        for _ in range(rounds):
            AnnoyIndex._lib = staticmethod(lambda: baseline)
            bests["serial"].append(timed(index, repeat))
            for workers, lib in variants.items():
                AnnoyIndex._lib = staticmethod(lambda lib=lib: lib)
                bests[workers].append(timed(index, repeat))
        row = {key: min(value) for key, value in bests.items()}
        best = min(row[w] for w in variant_workers)
        cells = f"| {n:,} |"
        for workers in variant_workers:
            cells += f" {row[workers] * 1e3:.2f} ({row[workers] / row['serial']:.2f}x) |"
        cells += f" {best / row['serial']:.2f}x |"
        print(cells, flush=True)


if __name__ == "__main__":
    main()
