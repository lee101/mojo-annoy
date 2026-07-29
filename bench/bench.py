"""Benchmark mojo-annoy against upstream Annoy through the same Python API."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import math
import os
import pathlib
import platform
import sys
import time

import numpy as np

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"),
)

from annoy import AnnoyIndex  # noqa: E402


def load_upstream():
    distribution = importlib.metadata.distribution("annoy")
    package = pathlib.Path(distribution.locate_file("annoy"))
    spec = importlib.util.spec_from_file_location(
        "bench_upstream_annoy",
        package / "__init__.py",
        submodule_search_locations=[str(package)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["bench_upstream_annoy"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.AnnoyIndex


UpstreamAnnoyIndex = load_upstream()


def cpu_name() -> str:
    try:
        for line in pathlib.Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def best_time(function, repeat: int = 3) -> float:
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def populate(cls, data: np.ndarray, metric: str):
    index = cls(data.shape[1], metric)
    index.set_seed(42)
    for i, row in enumerate(data):
        index.add_item(i, row)
    return index


def time_build(cls, data: np.ndarray, metric: str, trees: int) -> float:
    timings = []
    for _ in range(3):
        index = populate(cls, data, metric)
        start = time.perf_counter()
        index.build(trees, n_jobs=1)
        timings.append(time.perf_counter() - start)
    return min(timings)


def main() -> None:
    rng = np.random.default_rng(2026)
    build_data = rng.normal(size=(25_000, 32)).astype(np.float32)
    query_data = rng.normal(size=(1_000, 32)).astype(np.float32)

    ours_build = time_build(AnnoyIndex, build_data, "angular", 10)
    upstream_build = time_build(UpstreamAnnoyIndex, build_data, "angular", 10)

    ours_angular = populate(AnnoyIndex, build_data, "angular")
    upstream_angular = populate(UpstreamAnnoyIndex, build_data, "angular")
    ours_angular.build(10, n_jobs=1)
    upstream_angular.build(10, n_jobs=1)

    def query_all(index, search_k=-1):
        for query in query_data:
            index.get_nns_by_vector(query, 10, search_k)

    query_all(ours_angular)
    query_all(upstream_angular)
    ours_query = best_time(lambda: query_all(ours_angular))
    upstream_query = best_time(lambda: query_all(upstream_angular))
    ours_deep = best_time(lambda: query_all(ours_angular, 1000))
    upstream_deep = best_time(lambda: query_all(upstream_angular, 1000))

    euclidean_data = build_data[:10_000]
    euclidean_queries = query_data[:500]
    ours_euclidean = populate(AnnoyIndex, euclidean_data, "euclidean")
    upstream_euclidean = populate(UpstreamAnnoyIndex, euclidean_data, "euclidean")
    ours_euclidean.build(10, n_jobs=1)
    upstream_euclidean.build(10, n_jobs=1)

    def query_euclidean(index):
        for query in euclidean_queries:
            index.get_nns_by_vector(query, 10)

    query_euclidean(ours_euclidean)
    query_euclidean(upstream_euclidean)
    ours_euclidean_time = best_time(lambda: query_euclidean(ours_euclidean))
    upstream_euclidean_time = best_time(lambda: query_euclidean(upstream_euclidean))

    cases = [
        ("Build angular, 25k x 32, 10 trees", ours_build, upstream_build),
        ("1k angular queries, k=10, default search", ours_query, upstream_query),
        ("1k angular queries, k=10, search_k=1000", ours_deep, upstream_deep),
        ("500 euclidean queries, k=10, default", ours_euclidean_time, upstream_euclidean_time),
    ]

    print(f"Machine: {cpu_name()}; {platform.system()} {platform.machine()}; single process")
    print()
    print("| case | mojo-annoy | upstream annoy | upstream / Mojo |")
    print("|---|---:|---:|---:|")
    for name, ours, upstream in cases:
        print(
            f"| {name} | {ours * 1e3:.2f} ms | {upstream * 1e3:.2f} ms | "
            f"{upstream / ours:.2f}x |"
        )


if __name__ == "__main__":
    main()
