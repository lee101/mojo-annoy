"""Isolated cost of the build's inner projection pass (no trees, no traversal).

Times the same `projection` loop that dominates `mann_build_forest`, once on the
build-order `perm` (random row order, as the tree walks it) and once on a
sequential access order, at a fixed item count. This separates memory-pattern
cost from flop cost. Not part of the test suite.

Usage: `pixi run bench-project`
"""

from __future__ import annotations

import pathlib
import subprocess
import sys
import tempfile
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
SOURCE = (ROOT / "src" / "capi.mojo").read_text()

PROBE = '''
@export("probe_project")
def probe_project(
    vectors_addr: Int,
    perm_addr: Int,
    out_addr: Int,
    normal_addr: Int,
    count: Int,
    d: Int,
) abi("C") -> Int:
    var vectors = FPtr(unsafe_from_address=vectors_addr)
    var perm = IPtr(unsafe_from_address=perm_addr)
    var out = FPtr(unsafe_from_address=out_addr)
    var normal = FPtr(unsafe_from_address=normal_addr)
    for pos in range(count):
        out[pos] = projection(vectors, Int(perm[pos]), normal, d)
    return 0
'''


def build_probe() -> pathlib.Path:
    directory = pathlib.Path(tempfile.mkdtemp(prefix="mann-probe-"))
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
    import ctypes

    library = ctypes.CDLL(str(build_probe()))
    library.probe_project.argtypes = [ctypes.c_int64] * 4 + [ctypes.c_int64] * 2
    library.probe_project.restype = ctypes.c_int64

    d = 32
    n = 25_000
    rng = np.random.default_rng(2026)
    vectors = np.ascontiguousarray(rng.normal(size=(n, d)))
    normal = np.ascontiguousarray(rng.normal(size=d))
    normal /= np.linalg.norm(normal)
    out = np.empty(n, dtype=np.float64)
    orders = {
        "build order (random rows, tree perm)": rng.permutation(n),
        "sequential rows": np.arange(n),
    }
    for label, perm in orders.items():
        perm = np.ascontiguousarray(perm, dtype=np.int64)
        best = float("inf")
        for _ in range(9):
            start = time.perf_counter()
            library.probe_project(
                vectors.ctypes.data,
                perm.ctypes.data,
                out.ctypes.data,
                normal.ctypes.data,
                n,
                d,
            )
            best = min(best, time.perf_counter() - start)
        flops = 2 * n * d
        print(
            f"{label}: {best * 1e3:.2f} ms for {n:,} x {d} "
            f"-> {flops / best / 1e9:.1f} GFlop/s, "
            f"{n * d * 8 / best / 1e9:.1f} GB/s read"
        )


if __name__ == "__main__":
    sys.exit(main())
