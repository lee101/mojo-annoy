"""Random-projection forest construction, search, and distance kernels."""

from std.math import sqrt
from max.algorithm import sync_parallelize
from std.sys import simd_width_of

comptime FPtr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime F32Ptr = UnsafePointer[Float32, AnyOrigin[mut=True]]
comptime IPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]
comptime W = simd_width_of[DType.float64]()
comptime W32 = simd_width_of[DType.float32]()
comptime RERANK_PARALLEL_CANDIDATES = 4_096
comptime RERANK_PARALLEL_WORK = 262_144
comptime RERANK_WORKERS = 4


def fptr(addr: Int) -> FPtr:
    return FPtr(unsafe_from_address=addr)


def iptr(addr: Int) -> IPtr:
    return IPtr(unsafe_from_address=addr)


def f32ptr(addr: Int) -> F32Ptr:
    return F32Ptr(unsafe_from_address=addr)


def dot(a: FPtr, b: FPtr, n: Int) -> Float64:
    var acc0 = SIMD[DType.float64, W](0.0)
    var acc1 = SIMD[DType.float64, W](0.0)
    var acc2 = SIMD[DType.float64, W](0.0)
    var acc3 = SIMD[DType.float64, W](0.0)
    var i = 0
    while i + 4 * W <= n:
        acc0 += a.load[width=W](i) * b.load[width=W](i)
        acc1 += a.load[width=W](i + W) * b.load[width=W](i + W)
        acc2 += a.load[width=W](i + 2 * W) * b.load[width=W](i + 2 * W)
        acc3 += a.load[width=W](i + 3 * W) * b.load[width=W](i + 3 * W)
        i += 4 * W
    var total = (acc0 + acc1 + acc2 + acc3).reduce_add()
    while i + W <= n:
        total += (a.load[width=W](i) * b.load[width=W](i)).reduce_add()
        i += W
    while i < n:
        total += a[i] * b[i]
        i += 1
    return total


def dot32(a: F32Ptr, b: F32Ptr, n: Int) -> Float32:
    var acc0 = SIMD[DType.float32, W32](0.0)
    var acc1 = SIMD[DType.float32, W32](0.0)
    var acc2 = SIMD[DType.float32, W32](0.0)
    var acc3 = SIMD[DType.float32, W32](0.0)
    var i = 0
    while i + 4 * W32 <= n:
        acc0 += a.load[width=W32](i) * b.load[width=W32](i)
        acc1 += a.load[width=W32](i + W32) * b.load[width=W32](i + W32)
        acc2 += a.load[width=W32](i + 2 * W32) * b.load[width=W32](i + 2 * W32)
        acc3 += a.load[width=W32](i + 3 * W32) * b.load[width=W32](i + 3 * W32)
        i += 4 * W32
    var total = (acc0 + acc1 + acc2 + acc3).reduce_add()
    while i + W32 <= n:
        total += (a.load[width=W32](i) * b.load[width=W32](i)).reduce_add()
        i += W32
    while i < n:
        total += a[i] * b[i]
        i += 1
    return total


def metric_rank_distance32(
    a: F32Ptr,
    b: F32Ptr,
    d: Int,
    metric: Int,
    inv_norm_a: Float32,
    inv_norm_b: Float32,
) -> Float32:
    if metric == 0:
        if inv_norm_a == 0.0 and inv_norm_b == 0.0:
            return 0.0
        if inv_norm_a == 0.0 or inv_norm_b == 0.0:
            return 2.0
        var cosine = dot32(a, b, d) * inv_norm_a * inv_norm_b
        cosine = min(Float32(1.0), max(Float32(-1.0), cosine))
        return max(Float32(0.0), Float32(2.0) - Float32(2.0) * cosine)
    if metric == 1:
        var acc0 = SIMD[DType.float32, W32](0.0)
        var acc1 = SIMD[DType.float32, W32](0.0)
        var acc2 = SIMD[DType.float32, W32](0.0)
        var acc3 = SIMD[DType.float32, W32](0.0)
        var i = 0
        while i + 4 * W32 <= d:
            var delta0 = a.load[width=W32](i) - b.load[width=W32](i)
            var delta1 = a.load[width=W32](i + W32) - b.load[width=W32](i + W32)
            var delta2 = a.load[width=W32](i + 2 * W32) - b.load[width=W32](i + 2 * W32)
            var delta3 = a.load[width=W32](i + 3 * W32) - b.load[width=W32](i + 3 * W32)
            acc0 += delta0 * delta0
            acc1 += delta1 * delta1
            acc2 += delta2 * delta2
            acc3 += delta3 * delta3
            i += 4 * W32
        var total = (acc0 + acc1 + acc2 + acc3).reduce_add()
        while i + W32 <= d:
            var delta = a.load[width=W32](i) - b.load[width=W32](i)
            total += (delta * delta).reduce_add()
            i += W32
        while i < d:
            var delta = a[i] - b[i]
            total += delta * delta
            i += 1
        return total
    if metric == 2:
        var acc0 = SIMD[DType.float32, W32](0.0)
        var acc1 = SIMD[DType.float32, W32](0.0)
        var acc2 = SIMD[DType.float32, W32](0.0)
        var acc3 = SIMD[DType.float32, W32](0.0)
        var i = 0
        while i + 4 * W32 <= d:
            acc0 += abs(a.load[width=W32](i) - b.load[width=W32](i))
            acc1 += abs(a.load[width=W32](i + W32) - b.load[width=W32](i + W32))
            acc2 += abs(a.load[width=W32](i + 2 * W32) - b.load[width=W32](i + 2 * W32))
            acc3 += abs(a.load[width=W32](i + 3 * W32) - b.load[width=W32](i + 3 * W32))
            i += 4 * W32
        var total = (acc0 + acc1 + acc2 + acc3).reduce_add()
        while i + W32 <= d:
            total += abs(a.load[width=W32](i) - b.load[width=W32](i)).reduce_add()
            i += W32
        while i < d:
            total += abs(a[i] - b[i])
            i += 1
        return total
    if metric == 3:
        var total = Float32(0.0)
        for i in range(d):
            if Int64(a[i]) != Int64(b[i]):
                total += 1.0
        return total
    return -dot32(a, b, d)


def projection(vectors: FPtr, row: Int, normal: FPtr, d: Int) -> Float64:
    return dot(vectors + row * d, normal, d)


def metric_rank_distance(
    a: FPtr,
    b: FPtr,
    d: Int,
    metric: Int,
    inv_norm_a: Float64 = -1.0,
    inv_norm_b: Float64 = -1.0,
) -> Float64:
    if metric == 0:
        var inv_a = inv_norm_a
        var inv_b = inv_norm_b
        if inv_a < 0.0:
            var squared = dot(a, a, d)
            inv_a = 0.0 if squared == 0.0 else 1.0 / sqrt(squared)
        if inv_b < 0.0:
            var squared = dot(b, b, d)
            inv_b = 0.0 if squared == 0.0 else 1.0 / sqrt(squared)
        if inv_a == 0.0 and inv_b == 0.0:
            return 0.0
        if inv_a == 0.0 or inv_b == 0.0:
            return 2.0
        var cosine = dot(a, b, d) * inv_a * inv_b
        cosine = min(1.0, max(-1.0, cosine))
        return max(0.0, 2.0 - 2.0 * cosine)
    if metric == 1:
        var acc0 = SIMD[DType.float64, W](0.0)
        var acc1 = SIMD[DType.float64, W](0.0)
        var acc2 = SIMD[DType.float64, W](0.0)
        var acc3 = SIMD[DType.float64, W](0.0)
        var i = 0
        while i + 4 * W <= d:
            var delta0 = a.load[width=W](i) - b.load[width=W](i)
            var delta1 = a.load[width=W](i + W) - b.load[width=W](i + W)
            var delta2 = a.load[width=W](i + 2 * W) - b.load[width=W](i + 2 * W)
            var delta3 = a.load[width=W](i + 3 * W) - b.load[width=W](i + 3 * W)
            acc0 += delta0 * delta0
            acc1 += delta1 * delta1
            acc2 += delta2 * delta2
            acc3 += delta3 * delta3
            i += 4 * W
        var total = (acc0 + acc1 + acc2 + acc3).reduce_add()
        while i + W <= d:
            var delta = a.load[width=W](i) - b.load[width=W](i)
            total += (delta * delta).reduce_add()
            i += W
        while i < d:
            var delta = a[i] - b[i]
            total += delta * delta
            i += 1
        return total
    if metric == 2:
        var acc0 = SIMD[DType.float64, W](0.0)
        var acc1 = SIMD[DType.float64, W](0.0)
        var acc2 = SIMD[DType.float64, W](0.0)
        var acc3 = SIMD[DType.float64, W](0.0)
        var i = 0
        while i + 4 * W <= d:
            acc0 += abs(a.load[width=W](i) - b.load[width=W](i))
            acc1 += abs(a.load[width=W](i + W) - b.load[width=W](i + W))
            acc2 += abs(a.load[width=W](i + 2 * W) - b.load[width=W](i + 2 * W))
            acc3 += abs(a.load[width=W](i + 3 * W) - b.load[width=W](i + 3 * W))
            i += 4 * W
        var total = (acc0 + acc1 + acc2 + acc3).reduce_add()
        while i + W <= d:
            total += abs(a.load[width=W](i) - b.load[width=W](i)).reduce_add()
            i += W
        while i < d:
            total += abs(a[i] - b[i])
            i += 1
        return total
    if metric == 3:
        var total = 0.0
        for i in range(d):
            if Int64(a[i]) != Int64(b[i]):
                total += 1.0
        return total
    return -dot(a, b, d)


def metric_distance(a: FPtr, b: FPtr, d: Int, metric: Int) -> Float64:
    var distance = metric_rank_distance(a, b, d, metric)
    if metric == 0 or metric == 1:
        return sqrt(max(0.0, distance))
    return distance


def swap(values: IPtr, a: Int, b: Int):
    var tmp = values[a]
    values[a] = values[b]
    values[b] = tmp


def swap_float(values: FPtr, a: Int, b: Int):
    var tmp = values[a]
    values[a] = values[b]
    values[b] = tmp


def partition_median(perm: IPtr, values: FPtr, lo: Int, hi: Int, mid: Int):
    """Quickselect perm[lo:hi], placing the median at mid."""
    var left = lo
    var right = hi - 1
    while left < right:
        var pivot = values[(left + right) // 2]
        var i = left
        var j = right
        while i <= j:
            while values[i] < pivot:
                i += 1
            while values[j] > pivot:
                j -= 1
            if i <= j:
                swap(perm, i, j)
                swap_float(values, i, j)
                i += 1
                j -= 1
        if mid <= j:
            right = j
        elif mid >= i:
            left = i
        else:
            break


def make_normal(
    vectors: FPtr,
    vector_norms: FPtr,
    normal: FPtr,
    a: Int,
    b: Int,
    d: Int,
    metric: Int,
    fallback_dim: Int,
):
    var norm_a = 1.0
    var norm_b = 1.0
    if metric == 0 or metric == 4:
        norm_a = vector_norms[a]
        norm_b = vector_norms[b]
    var norm_acc = SIMD[DType.float64, W](0.0)
    var j = 0
    while j + W <= d:
        var value = (
            vectors.load[width=W](a * d + j) * norm_a
            - vectors.load[width=W](b * d + j) * norm_b
        )
        normal.store(j, value)
        norm_acc += value * value
        j += W
    var norm2 = norm_acc.reduce_add()
    while j < d:
        var value = vectors[a * d + j] * norm_a - vectors[b * d + j] * norm_b
        normal[j] = value
        norm2 += value * value
        j += 1
    if norm2 == 0.0:
        var zero = SIMD[DType.float64, W](0.0)
        j = 0
        while j + W <= d:
            normal.store(j, zero)
            j += W
        while j < d:
            normal[j] = 0.0
            j += 1
        normal[fallback_dim % d] = 1.0
    else:
        var inverse_norm = 1.0 / sqrt(norm2)
        var scale = SIMD[DType.float64, W](inverse_norm)
        j = 0
        while j + W <= d:
            normal.store(j, normal.load[width=W](j) * scale)
            j += W
        while j < d:
            normal[j] *= inverse_norm
            j += 1


@export("mann_build_forest")
def mann_build_forest(
    vectors_addr: Int,
    vector_norms_addr: Int,
    projections_addr: Int,
    perm_addr: Int,
    left_addr: Int,
    right_addr: Int,
    start_addr: Int,
    count_addr: Int,
    normal_addr: Int,
    threshold_addr: Int,
    node_count_addr: Int,
    n: Int,
    d: Int,
    trees: Int,
    max_nodes: Int,
    leaf_size: Int,
    metric: Int,
    seed: Int,
) abi("C") -> Int:
    var vectors = fptr(vectors_addr)
    var vector_norms = fptr(vector_norms_addr)
    var all_projections = fptr(projections_addr)
    var perms = iptr(perm_addr)
    var lefts = iptr(left_addr)
    var rights = iptr(right_addr)
    var starts = iptr(start_addr)
    var counts = iptr(count_addr)
    var normals = fptr(normal_addr)
    var thresholds = fptr(threshold_addr)
    var node_counts = iptr(node_count_addr)
    for tree in range(trees):
        var perm = perms + tree * n
        var projections = all_projections + tree * n
        for i in range(n):
            perm[i] = Int64(i)
        var node_base = tree * max_nodes
        lefts[node_base] = -1
        rights[node_base] = -1
        starts[node_base] = 0
        counts[node_base] = Int64(n)
        var used = 1
        var cursor = 0
        var state = UInt64(seed + tree * 104729 + 1)
        while cursor < used:
            var global_node = node_base + cursor
            var lo = Int(starts[global_node])
            var size = Int(counts[global_node])
            if size <= leaf_size:
                cursor += 1
                continue
            if used + 2 > max_nodes:
                return 0
            state = state * 6364136223846793005 + 1442695040888963407
            var pa = lo + Int(state >> 33) % size
            state = state * 6364136223846793005 + 1442695040888963407
            var pb = lo + Int(state >> 33) % size
            if pb == pa:
                pb = lo + (pb - lo + 1) % size
            var normal = normals + global_node * d
            make_normal(
                vectors,
                vector_norms,
                normal,
                Int(perm[pa]),
                Int(perm[pb]),
                d,
                metric,
                Int(state >> 17),
            )
            var mid = lo + size // 2
            for pos in range(lo, lo + size):
                projections[pos] = projection(
                    vectors, Int(perm[pos]), normal, d
                )
            partition_median(perm, projections, lo, lo + size, mid)
            thresholds[global_node] = projections[mid]
            var left_node = used
            var right_node = used + 1
            used += 2
            lefts[global_node] = Int64(left_node)
            rights[global_node] = Int64(right_node)
            lefts[node_base + left_node] = -1
            rights[node_base + left_node] = -1
            starts[node_base + left_node] = Int64(lo)
            counts[node_base + left_node] = Int64(mid - lo)
            lefts[node_base + right_node] = -1
            rights[node_base + right_node] = -1
            starts[node_base + right_node] = Int64(mid)
            counts[node_base + right_node] = Int64(lo + size - mid)
            cursor += 1
        node_counts[tree] = Int64(used)
    return 1


def heap_push(
    priorities: FPtr,
    nodes: IPtr,
    trees: IPtr,
    size: Int,
    priority: Float64,
    node: Int,
    tree: Int,
):
    var pos = size
    while pos > 0:
        var parent = (pos - 1) // 2
        if priorities[parent] >= priority:
            break
        priorities[pos] = priorities[parent]
        nodes[pos] = nodes[parent]
        trees[pos] = trees[parent]
        pos = parent
    priorities[pos] = priority
    nodes[pos] = Int64(node)
    trees[pos] = Int64(tree)


def rerank_range(
    vectors: F32Ptr,
    vector_norms: F32Ptr,
    query: F32Ptr,
    candidates: IPtr,
    candidate_dist: FPtr,
    begin: Int,
    end: Int,
    d: Int,
    metric: Int,
    query_inv_norm: Float32,
):
    for c in range(begin, end):
        var item = Int(candidates[c])
        candidate_dist[c] = Float64(metric_rank_distance32(
            vectors + item * d,
            query,
            d,
            metric,
            vector_norms[item],
            query_inv_norm,
        ))


@export("mann_query_forest")
def mann_query_forest(args_addr: Int) abi("C") -> Int:
    var args = iptr(args_addr)
    var vectors = f32ptr(Int(args[0]))
    var vector_norms = f32ptr(Int(args[1]))
    var query = f32ptr(Int(args[2]))
    var perms = iptr(Int(args[3]))
    var lefts = iptr(Int(args[4]))
    var rights = iptr(Int(args[5]))
    var starts = iptr(Int(args[6]))
    var counts = iptr(Int(args[7]))
    var normals = f32ptr(Int(args[8]))
    var thresholds = f32ptr(Int(args[9]))
    var present = iptr(Int(args[10]))
    var n = Int(args[11])
    var d = Int(args[12])
    var num_trees = Int(args[13])
    var max_nodes = Int(args[14])
    var metric = Int(args[15])
    var want = Int(args[16])
    var search_k = Int(args[17])
    var stamp = Int(args[18])
    var marks = iptr(Int(args[19]))
    var candidates = iptr(Int(args[20]))
    var candidate_dist = fptr(Int(args[21]))
    var result_ids = iptr(Int(args[22]))
    var result_dist = fptr(Int(args[23]))
    var heap_priority = fptr(Int(args[24]))
    var heap_node = iptr(Int(args[25]))
    var heap_tree = iptr(Int(args[26]))
    var heap_cap = Int(args[27])
    for i in range(want):
        result_ids[i] = -1
        result_dist[i] = 1.7976931348623157e308
    var heap_size = 0
    for tree in range(num_trees):
        heap_push(
            heap_priority,
            heap_node,
            heap_tree,
            heap_size,
            1.7976931348623157e308,
            0,
            tree,
        )
        heap_size += 1
    var candidate_count = 0
    var visited_count = 0
    while heap_size > 0 and visited_count < search_k:
        var priority = heap_priority[0]
        var node = Int(heap_node[0])
        var tree = Int(heap_tree[0])
        var new_size = heap_size - 1
        if new_size > 0:
            var replacement_priority = heap_priority[new_size]
            var replacement_node = heap_node[new_size]
            var replacement_tree = heap_tree[new_size]
            var pos = 0
            while True:
                var child = pos * 2 + 1
                if child >= new_size:
                    break
                if (
                    child + 1 < new_size
                    and heap_priority[child + 1] > heap_priority[child]
                ):
                    child += 1
                if heap_priority[child] <= replacement_priority:
                    break
                heap_priority[pos] = heap_priority[child]
                heap_node[pos] = heap_node[child]
                heap_tree[pos] = heap_tree[child]
                pos = child
            heap_priority[pos] = replacement_priority
            heap_node[pos] = replacement_node
            heap_tree[pos] = replacement_tree
        heap_size = new_size
        var global_node = tree * max_nodes + node
        if lefts[global_node] < 0:
            var begin = Int(starts[global_node])
            var size = Int(counts[global_node])
            var perm = perms + tree * n
            for pos in range(begin, begin + size):
                var item = Int(perm[pos])
                if present[item] != 0:
                    visited_count += 1
                    if marks[item] != Int64(stamp):
                        marks[item] = Int64(stamp)
                        candidates[candidate_count] = Int64(item)
                        candidate_count += 1
            continue
        var normal = normals + global_node * d
        var value = dot32(query, normal, d)
        var margin = value - thresholds[global_node]
        if heap_size < heap_cap:
            heap_push(
                heap_priority,
                heap_node,
                heap_tree,
                heap_size,
                min(priority, Float64(-margin)),
                Int(lefts[global_node]),
                tree,
            )
            heap_size += 1
        if heap_size < heap_cap:
            heap_push(
                heap_priority,
                heap_node,
                heap_tree,
                heap_size,
                min(priority, Float64(margin)),
                Int(rights[global_node]),
                tree,
            )
            heap_size += 1
    var query_inv_norm = Float32(-1.0)
    if metric == 0:
        var squared = dot32(query, query, d)
        query_inv_norm = 0.0 if squared == 0.0 else 1.0 / sqrt(squared)
    var parallel_rerank = (
        candidate_count >= RERANK_PARALLEL_CANDIDATES
        and candidate_count * d >= RERANK_PARALLEL_WORK
    )
    if parallel_rerank:
        def work(worker: Int) {var vectors, var vector_norms, var query, var candidates, var candidate_dist, var candidate_count, var d, var metric, var query_inv_norm}:
            var begin = worker * candidate_count // RERANK_WORKERS
            var end = (worker + 1) * candidate_count // RERANK_WORKERS
            rerank_range(
                vectors,
                vector_norms,
                query,
                candidates,
                candidate_dist,
                begin,
                end,
                d,
                metric,
                query_inv_norm,
            )
        sync_parallelize(work, RERANK_WORKERS)
    for c in range(candidate_count):
        var item = Int(candidates[c])
        var distance = candidate_dist[c] if parallel_rerank else Float64(metric_rank_distance32(
                vectors + item * d,
                query,
                d,
                metric,
                vector_norms[item],
                query_inv_norm,
            ))
        if distance > result_dist[want - 1]:
            continue
        var pos = want - 1
        while pos > 0:
            var prev_distance = result_dist[pos - 1]
            var prev_id = result_ids[pos - 1]
            if prev_distance < distance:
                break
            if prev_distance == distance and prev_id <= Int64(item):
                break
            result_dist[pos] = prev_distance
            result_ids[pos] = prev_id
            pos -= 1
        result_dist[pos] = distance
        result_ids[pos] = Int64(item)
    var found = min(candidate_count, want)
    if metric == 0 or metric == 1:
        for i in range(found):
            result_dist[i] = sqrt(max(0.0, result_dist[i]))
    return found


@export("mann_distance")
def mann_distance(
    a_addr: Int, b_addr: Int, d: Int, metric: Int
) abi("C") -> Float64:
    return metric_distance(fptr(a_addr), fptr(b_addr), d, metric)
