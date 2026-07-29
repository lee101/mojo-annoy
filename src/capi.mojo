"""Random-projection forest construction, search, and distance kernels."""

from std.math import sqrt
from std.algorithm import parallelize
from std.sys import simd_width_of

comptime FPtr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]
comptime W = simd_width_of[DType.float64]()


def fptr(addr: Int) -> FPtr:
    return FPtr(unsafe_from_address=addr)


def iptr(addr: Int) -> IPtr:
    return IPtr(unsafe_from_address=addr)


def dot(a: FPtr, b: FPtr, n: Int) -> Float64:
    var acc = SIMD[DType.float64, W](0.0)
    var i = 0
    while i + W <= n:
        acc += a.load[width=W](i) * b.load[width=W](i)
        i += W
    var total = acc.reduce_add()
    while i < n:
        total += a[i] * b[i]
        i += 1
    return total


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
        var acc = SIMD[DType.float64, W](0.0)
        var i = 0
        while i + W <= d:
            var delta = a.load[width=W](i) - b.load[width=W](i)
            acc += delta * delta
            i += W
        var total = acc.reduce_add()
        while i < d:
            var delta = a[i] - b[i]
            total += delta * delta
            i += 1
        return total
    if metric == 2:
        var acc = SIMD[DType.float64, W](0.0)
        var i = 0
        while i + W <= d:
            acc += abs(a.load[width=W](i) - b.load[width=W](i))
            i += W
        var total = acc.reduce_add()
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
        if priorities[parent] <= priority:
            break
        priorities[pos] = priorities[parent]
        nodes[pos] = nodes[parent]
        trees[pos] = trees[parent]
        pos = parent
    priorities[pos] = priority
    nodes[pos] = Int64(node)
    trees[pos] = Int64(tree)


@export("mann_query_forest")
def mann_query_forest(
    vectors_addr: Int,
    vector_norms_addr: Int,
    query_addr: Int,
    perm_addr: Int,
    left_addr: Int,
    right_addr: Int,
    start_addr: Int,
    count_addr: Int,
    normal_addr: Int,
    threshold_addr: Int,
    present_addr: Int,
    n: Int,
    d: Int,
    num_trees: Int,
    max_nodes: Int,
    metric: Int,
    want: Int,
    search_k: Int,
    stamp: Int,
    marks_addr: Int,
    candidates_addr: Int,
    candidate_dist_addr: Int,
    result_ids_addr: Int,
    result_dist_addr: Int,
    heap_priority_addr: Int,
    heap_node_addr: Int,
    heap_tree_addr: Int,
    heap_cap: Int,
) abi("C") -> Int:
    var vectors = fptr(vectors_addr)
    var vector_norms = fptr(vector_norms_addr)
    var query = fptr(query_addr)
    var perms = iptr(perm_addr)
    var lefts = iptr(left_addr)
    var rights = iptr(right_addr)
    var starts = iptr(start_addr)
    var counts = iptr(count_addr)
    var normals = fptr(normal_addr)
    var thresholds = fptr(threshold_addr)
    var present = iptr(present_addr)
    var marks = iptr(marks_addr)
    var candidates = iptr(candidates_addr)
    var candidate_dist = fptr(candidate_dist_addr)
    var result_ids = iptr(result_ids_addr)
    var result_dist = fptr(result_dist_addr)
    var heap_priority = fptr(heap_priority_addr)
    var heap_node = iptr(heap_node_addr)
    var heap_tree = iptr(heap_tree_addr)
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
            0.0,
            0,
            tree,
        )
        heap_size += 1
    var candidate_count = 0
    while heap_size > 0 and candidate_count < search_k:
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
                    and heap_priority[child + 1] < heap_priority[child]
                ):
                    child += 1
                if heap_priority[child] >= replacement_priority:
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
                if present[item] != 0 and marks[item] != Int64(stamp):
                    marks[item] = Int64(stamp)
                    candidates[candidate_count] = Int64(item)
                    candidate_count += 1
            continue
        var normal = normals + global_node * d
        var value = dot(query, normal, d)
        var margin = abs(value - thresholds[global_node])
        var near_node = Int(lefts[global_node])
        var far_node = Int(rights[global_node])
        if value > thresholds[global_node]:
            near_node = Int(rights[global_node])
            far_node = Int(lefts[global_node])
        if heap_size < heap_cap:
            heap_push(
                heap_priority,
                heap_node,
                heap_tree,
                heap_size,
                priority,
                near_node,
                tree,
            )
            heap_size += 1
        if heap_size < heap_cap:
            heap_push(
                heap_priority,
                heap_node,
                heap_tree,
                heap_size,
                priority + margin,
                far_node,
                tree,
            )
            heap_size += 1
    var query_inv_norm = -1.0
    if metric == 0:
        var squared = dot(query, query, d)
        query_inv_norm = 0.0 if squared == 0.0 else 1.0 / sqrt(squared)
    if candidate_count >= 4096 and candidate_count * d >= 262144:
        @parameter
        def score_candidate(c: Int):
            var item = Int(candidates[c])
            candidate_dist[c] = metric_rank_distance(
                vectors + item * d,
                query,
                d,
                metric,
                vector_norms[item],
                query_inv_norm,
            )

        parallelize[score_candidate](candidate_count, 4)
    for c in range(candidate_count):
        var item = Int(candidates[c])
        var distance = candidate_dist[c]
        if candidate_count < 4096 or candidate_count * d < 262144:
            distance = metric_rank_distance(
                vectors + item * d,
                query,
                d,
                metric,
                vector_norms[item],
                query_inv_norm,
            )
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
