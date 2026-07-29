"""ctypes bindings for the Mojo random-projection forest kernels."""

from __future__ import annotations

import ctypes
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.path.join(ROOT, "dist", "libmojo-annoy.so")

I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "mann_build_forest": ([I] * 18, I),
    "mann_query_forest": ([I] * 28, I),
    "mann_distance": ([I, I, I, I], F),
}

_library: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        if not os.path.exists(LIB):
            raise RuntimeError("Mojo library not built; run `pixi run build`")
        _library = ctypes.CDLL(LIB)
        for name, (argtypes, restype) in _SIGNATURES.items():
            function = getattr(_library, name)
            function.argtypes = argtypes
            function.restype = restype
    return _library


def addr(array: np.ndarray) -> int:
    if not isinstance(array, np.ndarray):
        raise TypeError("FFI buffers must be NumPy arrays")
    if array.dtype not in (np.dtype(np.float64), np.dtype(np.int64)):
        raise TypeError("FFI buffers must use float64 or int64 elements")
    if not array.flags.c_contiguous:
        raise ValueError("FFI buffers must be C-contiguous")
    if array.size == 0 or array.ctypes.data == 0:
        raise ValueError("FFI buffers must be non-empty and non-null")
    return int(array.ctypes.data)
