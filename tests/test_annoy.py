from __future__ import annotations

import importlib.metadata
import importlib.util
import pathlib
import sys

import numpy as np
import pytest

from annoy import AnnoyIndex
from annoy._lib import addr


def _load_upstream():
    distribution = importlib.metadata.distribution("annoy")
    package = pathlib.Path(distribution.locate_file("annoy"))
    spec = importlib.util.spec_from_file_location(
        "upstream_annoy",
        package / "__init__.py",
        submodule_search_locations=[str(package)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["upstream_annoy"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


UpstreamAnnoyIndex = _load_upstream().AnnoyIndex


def test_ffi_address_rejects_incompatible_buffers():
    with pytest.raises(TypeError):
        addr(np.ones(3, dtype=np.int32))
    with pytest.raises(ValueError):
        addr(np.ones((2, 2), dtype=np.float64)[:, 0])
    with pytest.raises(ValueError):
        addr(np.empty(0, dtype=np.int64))


def _pair(metric: str, data: np.ndarray, trees: int = 5):
    ours = AnnoyIndex(data.shape[1], metric)
    upstream = UpstreamAnnoyIndex(data.shape[1], metric)
    for i, row in enumerate(data):
        ours.add_item(i, row)
        upstream.add_item(i, row)
    ours.set_seed(17)
    upstream.set_seed(17)
    assert ours.build(trees, n_jobs=1)
    assert upstream.build(trees, n_jobs=1)
    return ours, upstream


@pytest.mark.parametrize("metric", ["angular", "euclidean", "manhattan", "dot"])
def test_distance_matches_upstream(metric):
    rng = np.random.default_rng(4)
    data = rng.normal(size=(24, 11))
    data[0] = 0
    ours, upstream = _pair(metric, data)
    for i, j in rng.integers(0, len(data), size=(40, 2)):
        assert ours.get_distance(i, j) == pytest.approx(
            upstream.get_distance(i, j), rel=2e-6, abs=2e-6
        )


def test_hamming_distance_matches_upstream():
    rng = np.random.default_rng(8)
    data = rng.integers(0, 2, size=(40, 67))
    ours, upstream = _pair("hamming", data)
    for i, j in rng.integers(0, len(data), size=(40, 2)):
        assert ours.get_distance(i, j) == upstream.get_distance(i, j)


@pytest.mark.parametrize("metric", ["angular", "euclidean", "manhattan"])
def test_simd_tail_distance_matches_upstream(metric):
    rng = np.random.default_rng(81)
    data = rng.normal(size=(20, 7))
    ours, upstream = _pair(metric, data, trees=2)
    for i, j in rng.integers(0, len(data), size=(20, 2)):
        assert ours.get_distance(i, j) == pytest.approx(
            upstream.get_distance(i, j), rel=2e-6, abs=2e-6
        )


def test_float32_query_is_zero_copy_and_conversion_buffer_is_reused():
    rng = np.random.default_rng(82)
    data = rng.normal(size=(100, 13))
    index, _ = _pair("euclidean", data, trees=2)
    query32 = np.ascontiguousarray(data[7], dtype=np.float32)
    index._query_vector.fill(np.nan)
    assert index.get_nns_by_vector(query32, 5, search_k=1000)[0] == 7
    assert np.isnan(index._query_vector).all()
    query64 = np.ascontiguousarray(data[8], dtype=np.float64)
    assert index.get_nns_by_vector(query64, 5, search_k=1000)[0] == 8
    assert np.array_equal(index._query_vector, query64.astype(np.float32))


def test_parallel_rerank_threshold_matches_exact_result():
    rng = np.random.default_rng(91)
    data = rng.normal(size=(4_103, 65))
    query = rng.normal(size=65)
    index = AnnoyIndex(65, "euclidean")
    for i, row in enumerate(data):
        index.add_item(i, row)
    index.build(1, n_jobs=1)
    got = index.get_nns_by_vector(query, 12, search_k=10**9)
    expected = np.argsort(np.sum((data - query) ** 2, axis=1))[:12].tolist()
    assert got == expected


@pytest.mark.parametrize(
    "metric", ["angular", "euclidean", "manhattan", "hamming", "dot"]
)
def test_exact_search_matches_upstream(metric):
    rng = np.random.default_rng(19)
    if metric == "hamming":
        data = rng.integers(0, 2, size=(257, 29))
        query = rng.integers(0, 2, size=29)
    else:
        data = rng.normal(size=(257, 13))
        query = rng.normal(size=13)
    ours, upstream = _pair(metric, data, trees=7)
    got_ids, got_distances = ours.get_nns_by_vector(
        query, 12, search_k=10**9, include_distances=True
    )
    ref_ids, ref_distances = upstream.get_nns_by_vector(
        query, 12, search_k=10**9, include_distances=True
    )
    if metric == "hamming":
        all_distances = np.count_nonzero(data != query, axis=1)
        cutoff = np.partition(all_distances, 11)[11]
        assert all(all_distances[item] <= cutoff for item in got_ids)
        assert got_distances == sorted(got_distances)
        assert got_distances == pytest.approx(ref_distances)
    else:
        assert got_ids == ref_ids
        assert got_distances == pytest.approx(ref_distances, rel=3e-6, abs=3e-6)


def test_default_search_has_upstream_quality():
    rng = np.random.default_rng(42)
    data = rng.normal(size=(4_000, 20))
    queries = rng.normal(size=(30, 20))
    ours, upstream = _pair("angular", data, trees=10)
    norms = np.linalg.norm(data, axis=1)
    ours_hits = 0
    upstream_hits = 0
    for query in queries:
        exact = set(
            np.argpartition(
                -(data @ query) / (norms * np.linalg.norm(query)), 10
            )[:10]
        )
        ours_hits += len(set(ours.get_nns_by_vector(query, 10)) & exact)
        upstream_hits += len(set(upstream.get_nns_by_vector(query, 10)) & exact)
    assert ours_hits / 300 >= 0.25
    assert ours_hits / 300 >= upstream_hits / 300 - 0.10


def test_sparse_ids_are_zero_filled_like_upstream():
    ours = AnnoyIndex(3, "euclidean")
    upstream = UpstreamAnnoyIndex(3, "euclidean")
    ours.add_item(3, [1, 2, 3])
    upstream.add_item(3, [1, 2, 3])
    assert ours.get_n_items() == upstream.get_n_items() == 4
    assert ours.get_item_vector(1) == upstream.get_item_vector(1) == [0.0, 0.0, 0.0]
    ours.build(2)
    upstream.build(2)
    got_ids, got_distances = ours.get_nns_by_item(1, 4, 1000, True)
    ref_ids, ref_distances = upstream.get_nns_by_item(1, 4, 1000, True)
    assert got_ids == ref_ids
    assert got_distances == pytest.approx(ref_distances)


def test_item_query_includes_itself_and_returns_lists():
    data = np.arange(60, dtype=float).reshape(20, 3)
    index, _ = _pair("euclidean", data)
    ids, distances = index.get_nns_by_item(7, 5, include_distances=True)
    assert isinstance(ids, list) and isinstance(distances, list)
    assert ids[0] == 7
    assert distances[0] == 0.0


def test_default_k_one_reaches_a_leaf_in_deep_tree():
    rng = np.random.default_rng(31)
    data = rng.normal(size=(4_000, 12))
    index = AnnoyIndex(12, "angular")
    for i, row in enumerate(data):
        index.add_item(i, row)
    index.build(10)
    result = index.get_nns_by_vector(data[123], 1)
    assert result == [123]


def test_seed_reproduces_forest_and_results():
    rng = np.random.default_rng(11)
    data = rng.normal(size=(300, 9))
    indexes = []
    for _ in range(2):
        index = AnnoyIndex(9, "angular")
        for i, row in enumerate(data):
            index.add_item(i, row)
        index.set_seed(123)
        index.build(4)
        indexes.append(index)
    assert np.array_equal(indexes[0]._perm, indexes[1]._perm)
    assert indexes[0].get_nns_by_vector(data[100], 10) == indexes[
        1
    ].get_nns_by_vector(data[100], 10)


def test_verbose_accepts_upstream_compatible_flag():
    index = AnnoyIndex(2)
    assert index.verbose(True) is None
    assert index._verbose is True
    assert index.verbose(False) is None
    assert index._verbose is False


def test_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(2)
    data = rng.normal(size=(211, 8))
    index, _ = _pair("manhattan", data)
    before = index.get_nns_by_vector(data[17], 20, 500, True)
    filename = tmp_path / "forest.ann"
    assert index.save(filename, prefault=True)
    loaded = AnnoyIndex(8, "manhattan")
    assert loaded.load(filename, prefault=True)
    assert loaded.get_n_items() == 211
    assert loaded.get_n_trees() == 5
    assert loaded.get_nns_by_vector(data[17], 20, 500, True) == before


@pytest.mark.parametrize("corruption", ["shape", "dtype", "link"])
def test_load_rejects_unsafe_forest_data(tmp_path, corruption):
    index = AnnoyIndex(3, "euclidean")
    for i in range(40):
        index.add_item(i, [i, i + 1, i + 2])
    index.build(2)
    good = tmp_path / "good.ann"
    bad = tmp_path / "bad.ann"
    index.save(good)
    with np.load(good, allow_pickle=False) as archive:
        payload = {name: archive[name] for name in archive.files}
    if corruption == "shape":
        payload["normals"] = payload["normals"][:, :, :-1]
    elif corruption == "dtype":
        payload["left"] = payload["left"].astype(np.uint64)
    else:
        payload["left"] = payload["left"].copy()
        payload["left"][0, 0] = payload["node_count"][0]
    with open(bad, "wb") as stream:
        np.savez(stream, **payload)
    with pytest.raises(OSError):
        AnnoyIndex(3, "euclidean").load(bad)


def test_on_disk_build_writes_loadable_index(tmp_path):
    filename = tmp_path / "ondisk.ann"
    index = AnnoyIndex(2, "euclidean")
    assert index.on_disk_build(filename)
    index.add_item(0, [0, 0])
    index.add_item(1, [1, 1])
    index.build(2)
    assert filename.is_file()
    loaded = AnnoyIndex(2, "euclidean")
    assert loaded.load(filename)
    assert loaded.get_nns_by_item(0, 2, 100) == [0, 1]


def test_unbuild_rebuild_and_unload():
    index = AnnoyIndex(2, "euclidean")
    for i in range(20):
        index.add_item(i, [i, -i])
    index.build(2)
    assert index.get_n_trees() == 2
    assert index.unbuild()
    assert not index.unbuild()
    index.add_item(20, [20, -20])
    index.build(3)
    assert index.get_n_items() == 21
    assert index.get_n_trees() == 3
    assert index.unload()
    assert index.get_n_items() == 0
    assert not index.unload()


def test_validation_and_empty_result():
    with pytest.raises(ValueError):
        AnnoyIndex(0)
    with pytest.raises(ValueError):
        AnnoyIndex(2, "cosine")
    index = AnnoyIndex(2)
    with pytest.raises(IndexError):
        index.add_item(0, [1])
    with pytest.raises(TypeError):
        index.add_item(0, [1 + 2j, 0])
    with pytest.raises(RuntimeError):
        index.build(2)
    index.add_item(0, [1, 0])
    index.build(1)
    with pytest.raises(TypeError):
        index.get_nns_by_vector([1 + 2j, 0], 1)
    assert index.get_nns_by_vector([1, 0], 0) == []
    assert index.get_nns_by_vector([1, 0], 0, include_distances=True) == ([], [])
