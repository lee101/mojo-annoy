"""Annoy-compatible random-projection forest index."""

from __future__ import annotations

import os
from typing import Iterable

import numpy as np

from ._lib import addr, lib

_METRICS = {
    "angular": 0,
    "euclidean": 1,
    "manhattan": 2,
    "hamming": 3,
    "dot": 4,
}


class AnnoyIndex:
    def __init__(self, f: int, metric: str = "angular") -> None:
        if int(f) <= 0:
            raise ValueError("f must be positive")
        if metric not in _METRICS:
            raise ValueError(f"unknown metric {metric!r}")
        self.f = int(f)
        self.metric = metric
        self._metric_code = _METRICS[metric]
        self._items: dict[int, np.ndarray] = {}
        self._vectors: np.ndarray | None = None
        self._built = False
        self._n_trees = 0
        self._seed = 0
        self._verbose = False
        self._on_disk_path: str | None = None
        self._stamp = 0

    def add_item(self, i: int, vector: Iterable[float]) -> None:
        if self._built:
            raise RuntimeError("You can't add an item to a built index")
        item = int(i)
        if item < 0:
            raise IndexError("item index must be non-negative")
        value = self._as_vector(vector)
        if value.ndim != 1 or value.size != self.f:
            raise IndexError(f"Vector has wrong length (expected {self.f}, got {value.size})")
        self._items[item] = value.copy()

    def build(self, n_trees: int, n_jobs: int = -1) -> bool:
        del n_jobs
        if self._built:
            raise RuntimeError("You can't build a built index")
        if not self._items:
            raise RuntimeError("You can't build an empty index")
        trees = int(n_trees)
        if trees <= 0:
            raise ValueError("n_trees must be positive")
        n = max(self._items) + 1
        vectors = np.zeros((n, self.f), dtype=np.float64)
        self._present = np.zeros(n, dtype=np.int64)
        for item, vector in self._items.items():
            vectors[item] = vector
            self._present[item] = 1
        self._n_active = len(self._items)
        leaf_size = 32
        max_nodes = max(1, 4 * ((n + leaf_size - 1) // leaf_size) + 8)
        shape = (trees, max_nodes)
        self._vectors = vectors
        norm_sq = np.einsum("ij,ij->i", vectors, vectors)
        self._inv_norm = np.zeros_like(norm_sq)
        np.sqrt(norm_sq, out=self._inv_norm)
        np.reciprocal(
            self._inv_norm,
            out=self._inv_norm,
            where=self._inv_norm != 0.0,
        )
        self._perm = np.empty((trees, n), dtype=np.int64)
        self._left = np.empty(shape, dtype=np.int64)
        self._right = np.empty(shape, dtype=np.int64)
        self._start = np.empty(shape, dtype=np.int64)
        self._count = np.empty(shape, dtype=np.int64)
        self._normals = np.zeros((trees, max_nodes, self.f), dtype=np.float64)
        self._threshold = np.zeros(shape, dtype=np.float64)
        self._node_count = np.empty(trees, dtype=np.int64)
        projections = np.empty((trees, n), dtype=np.float64)
        ok = lib().mann_build_forest(
            addr(self._vectors),
            addr(self._inv_norm),
            addr(projections),
            addr(self._perm),
            addr(self._left),
            addr(self._right),
            addr(self._start),
            addr(self._count),
            addr(self._normals),
            addr(self._threshold),
            addr(self._node_count),
            n,
            self.f,
            trees,
            max_nodes,
            leaf_size,
            self._metric_code,
            self._seed,
        )
        if not ok:
            self._clear_forest()
            raise RuntimeError("forest node capacity exhausted")
        self._built = True
        self._n_trees = trees
        self._max_nodes = max_nodes
        self._marks = np.zeros(n, dtype=np.int64)
        self._candidates = np.empty(n, dtype=np.int64)
        self._total_nodes = int(self._node_count.sum())
        self._init_search_arrays()
        self._init_query_scratch()
        if self._on_disk_path is not None:
            self.save(self._on_disk_path)
        return True

    def unbuild(self) -> bool:
        if not self._built:
            return False
        if self._vectors is not None:
            self._items = {
                i: self._vectors[i].copy()
                for i in range(len(self._vectors))
                if self._present[i]
            }
        self._clear_forest()
        return True

    def save(self, fn: os.PathLike[str] | str, prefault: bool = False) -> bool:
        del prefault
        self._require_built()
        with open(os.fspath(fn), "wb") as stream:
            np.savez(
                stream,
                format_version=np.array([1], dtype=np.int64),
                f=np.array([self.f], dtype=np.int64),
                metric=np.array([self.metric]),
                n_trees=np.array([self._n_trees], dtype=np.int64),
                max_nodes=np.array([self._max_nodes], dtype=np.int64),
                seed=np.array([self._seed], dtype=np.int64),
                vectors=self._vectors,
                inv_norm=self._inv_norm,
                present=self._present,
                perm=self._perm,
                left=self._left,
                right=self._right,
                start=self._start,
                count=self._count,
                normals=self._normals,
                threshold=self._threshold,
                node_count=self._node_count,
            )
        return True

    def load(self, fn: os.PathLike[str] | str, prefault: bool = False) -> bool:
        del prefault
        with np.load(os.fspath(fn), allow_pickle=False) as data:
            required = {
                "format_version",
                "f",
                "metric",
                "n_trees",
                "max_nodes",
                "seed",
                "vectors",
                "inv_norm",
                "present",
                "perm",
                "left",
                "right",
                "start",
                "count",
                "normals",
                "threshold",
                "node_count",
            }
            if not required.issubset(data.files):
                raise OSError("incomplete mojo-annoy index")
            if data["format_version"].shape != (1,) or int(data["format_version"][0]) != 1:
                raise OSError("unsupported mojo-annoy index format")
            if (
                data["f"].shape != (1,)
                or data["metric"].shape != (1,)
                or int(data["f"][0]) != self.f
                or str(data["metric"][0]) != self.metric
            ):
                raise OSError("index dimension or metric does not match constructor")
            if (
                data["n_trees"].shape != (1,)
                or data["max_nodes"].shape != (1,)
                or data["seed"].shape != (1,)
            ):
                raise OSError("invalid mojo-annoy index metadata")
            integer_arrays = (
                "n_trees",
                "max_nodes",
                "seed",
                "present",
                "perm",
                "left",
                "right",
                "start",
                "count",
                "node_count",
            )
            float_arrays = ("vectors", "inv_norm", "normals", "threshold")
            if any(data[name].dtype != np.int64 for name in integer_arrays) or any(
                data[name].dtype != np.float64 for name in float_arrays
            ):
                raise OSError("invalid array dtype in mojo-annoy index")
            n_trees = int(data["n_trees"][0])
            max_nodes = int(data["max_nodes"][0])
            seed = int(data["seed"][0])
            arrays = {
                "vectors": np.ascontiguousarray(data["vectors"], dtype=np.float64),
                "inv_norm": np.ascontiguousarray(data["inv_norm"], dtype=np.float64),
                "present": np.ascontiguousarray(data["present"], dtype=np.int64),
                "perm": np.ascontiguousarray(data["perm"], dtype=np.int64),
                "left": np.ascontiguousarray(data["left"], dtype=np.int64),
                "right": np.ascontiguousarray(data["right"], dtype=np.int64),
                "start": np.ascontiguousarray(data["start"], dtype=np.int64),
                "count": np.ascontiguousarray(data["count"], dtype=np.int64),
                "normals": np.ascontiguousarray(data["normals"], dtype=np.float64),
                "threshold": np.ascontiguousarray(data["threshold"], dtype=np.float64),
                "node_count": np.ascontiguousarray(data["node_count"], dtype=np.int64),
            }
        self._validate_loaded(n_trees, max_nodes, arrays)
        self._vectors = arrays["vectors"]
        self._inv_norm = arrays["inv_norm"]
        self._present = arrays["present"]
        self._n_active = int(self._present.sum())
        self._perm = arrays["perm"]
        self._left = arrays["left"]
        self._right = arrays["right"]
        self._start = arrays["start"]
        self._count = arrays["count"]
        self._normals = arrays["normals"]
        self._threshold = arrays["threshold"]
        self._node_count = arrays["node_count"]
        self._n_trees = n_trees
        self._max_nodes = max_nodes
        self._seed = seed
        self._items = {}
        self._built = True
        self._marks = np.zeros(len(self._vectors), dtype=np.int64)
        self._candidates = np.empty(len(self._vectors), dtype=np.int64)
        self._total_nodes = int(self._node_count.sum())
        self._init_search_arrays()
        self._init_query_scratch()
        return True

    def unload(self) -> bool:
        had_data = self._built or bool(self._items)
        self._items = {}
        self._vectors = None
        self.__dict__.pop("_present", None)
        self.__dict__.pop("_n_active", None)
        self._clear_forest()
        return had_data

    def on_disk_build(self, fn: os.PathLike[str] | str) -> bool:
        if self._built:
            raise RuntimeError("You can't call on_disk_build on a built index")
        self._on_disk_path = os.fspath(fn)
        return True

    def set_seed(self, seed: int) -> None:
        if self._built:
            raise RuntimeError("seed must be set before build")
        self._seed = int(seed)

    def verbose(self, v: bool) -> None:
        self._verbose = bool(v)

    def get_n_items(self) -> int:
        if self._vectors is not None:
            return len(self._vectors)
        return max(self._items, default=-1) + 1

    def get_n_trees(self) -> int:
        return self._n_trees

    def get_item_vector(self, i: int) -> list[float]:
        item = self._check_item(i)
        if self._vectors is not None:
            return self._vectors[item].tolist()
        if item in self._items:
            return self._items[item].tolist()
        return [0.0] * self.f

    def get_distance(self, i: int, j: int) -> float:
        left = self._vector_for_item(i)
        right = self._vector_for_item(j)
        distance = float(
            lib().mann_distance(addr(left), addr(right), self.f, self._metric_code)
        )
        return -distance if self.metric == "dot" else distance

    def get_nns_by_item(
        self,
        i: int,
        n: int,
        search_k: int = -1,
        include_distances: bool = False,
    ):
        return self.get_nns_by_vector(
            self._vector_for_item(i), n, search_k, include_distances
        )

    def get_nns_by_vector(
        self,
        vector: Iterable[float],
        n: int,
        search_k: int = -1,
        include_distances: bool = False,
    ):
        self._require_built()
        value = np.asarray(vector)
        if value.ndim != 1 or value.size != self.f:
            raise IndexError(f"Vector has wrong length (expected {self.f}, got {value.size})")
        if value.dtype.kind == "c":
            raise TypeError("complex vector values are not supported")
        if value.dtype == np.float32 and value.flags.c_contiguous:
            query = value
        else:
            query = self._query_vector
            try:
                np.copyto(query, value, casting="unsafe")
            except (TypeError, ValueError, OverflowError) as error:
                raise TypeError("vector values must be real numbers") from error
        want = min(max(0, int(n)), self._n_active)
        if want == 0:
            return ([], []) if include_distances else []
        budget = int(search_k)
        if budget == -1:
            budget = want * self._n_trees
        if budget <= 0:
            budget = 1
        total_nodes = self._total_nodes
        budget = min(budget, self._n_active * self._n_trees)
        heap_cap = min(total_nodes, budget + self._n_trees + 128)
        result_ids = self._result_ids
        result_dist = self._result_dist
        self._stamp += 1
        if self._stamp == np.iinfo(np.int64).max:
            self._marks.fill(0)
            self._stamp = 1
        self._query_args[2] = addr(query)
        self._query_args[16] = want
        self._query_args[17] = budget
        self._query_args[18] = self._stamp
        self._query_args[27] = heap_cap
        found = lib().mann_query_forest(addr(self._query_args))
        ids = result_ids[:found].tolist()
        if not include_distances:
            return ids
        distances = result_dist[:found]
        if self.metric == "dot":
            distances = -distances
        return ids, distances.tolist()

    def _vector_for_item(self, i: int) -> np.ndarray:
        item = self._check_item(i)
        if self._vectors is not None:
            return self._vectors[item]
        if item in self._items:
            return self._items[item]
        return np.zeros(self.f, dtype=np.float64)

    def _check_item(self, i: int) -> int:
        item = int(i)
        if item < 0 or item >= self.get_n_items():
            raise IndexError("item index out of range")
        return item

    def _require_built(self) -> None:
        if not self._built or self._vectors is None:
            raise RuntimeError("index is not built")

    def _as_vector(self, vector: Iterable[float]) -> np.ndarray:
        value = np.asarray(vector)
        if value.ndim != 1 or value.size != self.f:
            raise IndexError(f"Vector has wrong length (expected {self.f}, got {value.size})")
        if value.dtype.kind == "c":
            raise TypeError("complex vector values are not supported")
        try:
            return np.ascontiguousarray(value, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as error:
            raise TypeError("vector values must be real numbers") from error

    def _validate_loaded(
        self, n_trees: int, max_nodes: int, arrays: dict[str, np.ndarray]
    ) -> None:
        vectors = arrays["vectors"]
        if vectors.ndim != 2 or vectors.shape[1:] != (self.f,) or vectors.shape[0] == 0:
            raise OSError("invalid vector shape in mojo-annoy index")
        n = vectors.shape[0]
        expected = {
            "inv_norm": (n,),
            "present": (n,),
            "perm": (n_trees, n),
            "left": (n_trees, max_nodes),
            "right": (n_trees, max_nodes),
            "start": (n_trees, max_nodes),
            "count": (n_trees, max_nodes),
            "normals": (n_trees, max_nodes, self.f),
            "threshold": (n_trees, max_nodes),
            "node_count": (n_trees,),
        }
        if n_trees <= 0 or max_nodes <= 0:
            raise OSError("invalid forest dimensions in mojo-annoy index")
        if any(arrays[name].shape != shape for name, shape in expected.items()):
            raise OSError("invalid array shape in mojo-annoy index")
        if not np.all((arrays["present"] == 0) | (arrays["present"] == 1)):
            raise OSError("invalid presence bitmap in mojo-annoy index")
        if not np.any(arrays["present"]):
            raise OSError("empty mojo-annoy index")
        if np.any(arrays["perm"] < 0) or np.any(arrays["perm"] >= n):
            raise OSError("invalid permutation in mojo-annoy index")
        for tree, used64 in enumerate(arrays["node_count"]):
            used = int(used64)
            if used <= 0 or used > max_nodes:
                raise OSError("invalid node count in mojo-annoy index")
            left = arrays["left"][tree, :used]
            right = arrays["right"][tree, :used]
            leaves = left < 0
            if np.any((left[~leaves] >= used) | (right[~leaves] < 0) | (right[~leaves] >= used)):
                raise OSError("invalid tree links in mojo-annoy index")
            starts = arrays["start"][tree, :used][leaves]
            counts = arrays["count"][tree, :used][leaves]
            if np.any(starts < 0) or np.any(counts < 0) or np.any(starts > n - counts):
                raise OSError("invalid leaf span in mojo-annoy index")

    def _init_query_scratch(self) -> None:
        self._heap_priority = np.empty(self._total_nodes, dtype=np.float64)
        self._heap_node = np.empty(self._total_nodes, dtype=np.int64)
        self._heap_tree = np.empty(self._total_nodes, dtype=np.int64)
        self._result_ids = np.empty(self._n_active, dtype=np.int64)
        self._result_dist = np.empty(self._n_active, dtype=np.float64)
        self._candidate_dist = np.empty(self._n_active, dtype=np.float64)
        self._query_vector = np.empty(self.f, dtype=np.float32)
        self._query_args = np.array(
            [
                addr(self._search_vectors),
                addr(self._search_inv_norm),
                0,
                addr(self._perm),
                addr(self._left),
                addr(self._right),
                addr(self._start),
                addr(self._count),
                addr(self._search_normals),
                addr(self._search_threshold),
                addr(self._present),
                self.get_n_items(),
                self.f,
                self._n_trees,
                self._max_nodes,
                self._metric_code,
                0,
                0,
                0,
                addr(self._marks),
                addr(self._candidates),
                addr(self._candidate_dist),
                addr(self._result_ids),
                addr(self._result_dist),
                addr(self._heap_priority),
                addr(self._heap_node),
                addr(self._heap_tree),
                0,
            ],
            dtype=np.int64,
        )

    def _init_search_arrays(self) -> None:
        self._search_vectors = np.ascontiguousarray(self._vectors, dtype=np.float32)
        self._search_inv_norm = np.ascontiguousarray(self._inv_norm, dtype=np.float32)
        self._search_normals = np.ascontiguousarray(self._normals, dtype=np.float32)
        self._search_threshold = np.ascontiguousarray(self._threshold, dtype=np.float32)

    def _clear_forest(self) -> None:
        self._built = False
        self._n_trees = 0
        for name in (
            "_perm",
            "_left",
            "_right",
            "_start",
            "_count",
            "_normals",
            "_threshold",
            "_node_count",
            "_marks",
            "_candidates",
            "_max_nodes",
            "_present",
            "_n_active",
            "_inv_norm",
            "_total_nodes",
            "_heap_priority",
            "_heap_node",
            "_heap_tree",
            "_result_ids",
            "_result_dist",
            "_candidate_dist",
            "_query_vector",
            "_query_args",
            "_search_vectors",
            "_search_inv_norm",
            "_search_normals",
            "_search_threshold",
        ):
            self.__dict__.pop(name, None)
