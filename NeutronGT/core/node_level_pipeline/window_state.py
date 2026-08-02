import os
import time
from collections import deque

import torch
import torch.distributed as dist
from torch_geometric.utils import subgraph

from .struct_info import StructInfo


def _empty_cache(feature: torch.Tensor, device: str):
    return feature.new_empty((0, feature.shape[1]), device=device)


def _window_state_dir(args):
    cache_dir = os.path.join(args.dataset_dir, args.dataset, 'window_state_cache')
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def _bundle_path(args, version: int, rank: int):
    run_id = getattr(args, 'sync_run_id', 'default')
    return os.path.join(_window_state_dir(args), f'window_state_{run_id}_v{version}_rank{rank}.pt')


def _done_path(args, version: int, rank: int):
    run_id = getattr(args, 'sync_run_id', 'default')
    return os.path.join(_window_state_dir(args), f'window_state_{run_id}_v{version}_rank{rank}.done')


def _wait_timeout_seconds(args) -> float:
    return max(float(getattr(args, 'distributed_timeout_minutes', 10)) * 60.0 * 4.0, 24 * 60 * 60.0)


def _wait_for_path(path: str, timeout_seconds: float, poll_seconds: float = 30.0):
    start = time.time()
    while not os.path.exists(path):
        if time.time() - start > timeout_seconds:
            raise TimeoutError(f"Timed out waiting for file: {path}")
        time.sleep(poll_seconds)


def _atomic_torch_save(obj, path: str):
    tmp_path = f"{path}.tmp.{os.getpid()}"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def _touch_done(path: str, timing_stats: dict):
    tmp_path = f"{path}.tmp.{os.getpid()}"
    with open(tmp_path, 'w') as f:
        f.write(repr(timing_stats))
        f.write('\n')
    os.replace(tmp_path, path)


def _stash_global_window_state_cpu(structInfo: StructInfo):
    if structInfo.wm.partitioned_results:
        structInfo.global_partitioned_results_cpu = structInfo.wm.partitioned_results
    if structInfo.wm.sub_edge_index_for_partition_results:
        structInfo.global_sub_edge_index_for_partition_results_cpu = structInfo.wm.sub_edge_index_for_partition_results


def restore_global_window_state(structInfo: StructInfo):
    if structInfo.global_partitioned_results_cpu is not None:
        structInfo.wm.partitioned_results = structInfo.global_partitioned_results_cpu
    if structInfo.global_sub_edge_index_for_partition_results_cpu is not None:
        structInfo.wm.sub_edge_index_for_partition_results = structInfo.global_sub_edge_index_for_partition_results_cpu


def _release_hot_global_window_state(structInfo: StructInfo):
    structInfo.wm.partitioned_results = []
    structInfo.wm.sub_edge_index_for_partition_results = []
    structInfo.wm.dup_nodes_per_partition = []
    structInfo.spatial_pos_by_pid = []


def _assign_local_window_bundle(structInfo: StructInfo, bundle):
    structInfo.local_partition_ids = bundle['local_partition_ids']
    structInfo.local_partitions = bundle['local_partitions']
    structInfo.local_dup_nodes_per_partition = bundle.get('local_dup_nodes_per_partition', [])
    structInfo.local_sub_edge_index_for_partition_results = bundle.get('local_sub_edge_index_list', [])


def _compute_local_duplicate_nodes(local_partitions):
    if not local_partitions:
        return [], []
    all_nodes = torch.cat([part.to(torch.long).cpu() for part in local_partitions]) if local_partitions else torch.empty((0,), dtype=torch.long)
    if all_nodes.numel() == 0:
        return [part.clone() for part in local_partitions], [torch.empty((0,), dtype=torch.long) for _ in local_partitions]

    unique_nodes, counts = torch.unique(all_nodes, return_counts=True, sorted=True)
    duplicated_nodes = unique_nodes[counts >= 2]
    if duplicated_nodes.numel() == 0:
        return [part.clone() for part in local_partitions], [torch.empty((0,), dtype=torch.long) for _ in local_partitions]

    reranged_partitions = []
    dup_nodes_per_partition = []
    for part in local_partitions:
        part_cpu = part.to(torch.long).cpu()
        dup_mask = torch.isin(part_cpu, duplicated_nodes)
        dup_nodes = part_cpu[dup_mask]
        non_dup_nodes = part_cpu[~dup_mask]
        reranged_partitions.append(torch.cat([dup_nodes, non_dup_nodes], dim=0))
        dup_nodes_per_partition.append(dup_nodes)
    return reranged_partitions, dup_nodes_per_partition


def _remap_sub_edge_index_for_reordered_partition(old_partition: torch.Tensor,
                                                  new_partition: torch.Tensor,
                                                  old_edge_index: torch.Tensor):
    old_edge_index = old_edge_index.to(torch.long).cpu()
    if old_edge_index.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long)

    old_partition = old_partition.to(torch.long).cpu()
    new_partition = new_partition.to(torch.long).cpu()
    if old_partition.numel() != new_partition.numel():
        raise RuntimeError(
            f"Partition remap size mismatch: old={old_partition.numel()}, new={new_partition.numel()}"
        )
    if torch.equal(old_partition, new_partition):
        return old_edge_index

    sorted_new_nodes, sorted_to_new_local = torch.sort(new_partition)
    edge_src_global = old_partition[old_edge_index[0]]
    edge_dst_global = old_partition[old_edge_index[1]]

    def lookup_new_local(edge_nodes_global: torch.Tensor):
        pos = torch.searchsorted(sorted_new_nodes, edge_nodes_global)
        in_bounds = pos < sorted_new_nodes.numel()
        matched = torch.zeros_like(in_bounds, dtype=torch.bool)
        if bool(in_bounds.any().item()):
            matched[in_bounds] = sorted_new_nodes[pos[in_bounds]] == edge_nodes_global[in_bounds]
        if not bool(matched.all().item()):
            raise RuntimeError("Cached sub-edge remap failed: new partition is not the same node set as old partition.")
        return sorted_to_new_local[pos].to(torch.long)

    new_src = lookup_new_local(edge_src_global)
    new_dst = lookup_new_local(edge_dst_global)
    return torch.stack([new_src, new_dst], dim=0)


def _build_local_ppr_sub_edge_index_list(structInfo: StructInfo, local_partitions):
    if structInfo.sorted_ppr_matrix is None:
        return []
    ppr_edge_index = structInfo.sorted_ppr_matrix[0].to('cpu')
    local_ppr_sub_edge_index_list = []
    for partition in local_partitions:
        local_edge_index, _ = subgraph(
            partition,
            ppr_edge_index,
            relabel_nodes=True,
            num_nodes=structInfo.num_nodes,
        )
        local_ppr_sub_edge_index_list.append(local_edge_index)
    return local_ppr_sub_edge_index_list


def _subgraph_from_csr(node_set: torch.Tensor, rowptr: torch.Tensor, col: torch.Tensor):
    node_set = node_set.to(torch.long).cpu()
    if node_set.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long)

    sort_order = torch.argsort(node_set)
    sorted_nodes = node_set[sort_order]
    row_chunks = []
    col_chunks = []
    for local_src, global_src in enumerate(node_set.tolist()):
        start = int(rowptr[global_src].item())
        end = int(rowptr[global_src + 1].item())
        neighbors = col[start:end].to(torch.long)
        if neighbors.numel() == 0:
            continue
        positions = torch.searchsorted(sorted_nodes, neighbors)
        in_bounds = positions < sorted_nodes.numel()
        matched = torch.zeros_like(in_bounds, dtype=torch.bool)
        if in_bounds.any():
            matched[in_bounds] = sorted_nodes[positions[in_bounds]] == neighbors[in_bounds]
        if not matched.any():
            continue
        local_dst = sort_order[positions[matched]].to(torch.long)
        row_chunks.append(torch.full((local_dst.numel(),), local_src, dtype=torch.long))
        col_chunks.append(local_dst)
    if not row_chunks:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.stack([torch.cat(row_chunks), torch.cat(col_chunks)], dim=0)


def _build_local_sub_edge_index_list(structInfo: StructInfo, local_partitions):
    local_sub_edge_index_list = []
    if structInfo.graph_edge_index is not None:
        edge_index = structInfo.graph_edge_index.to('cpu')
        for partition in local_partitions:
            local_edge_index, _ = subgraph(
                partition,
                edge_index,
                relabel_nodes=True,
                num_nodes=structInfo.num_nodes,
            )
            # torch_scatter 要求 int64 index; graph_edge_index 为 int32, subgraph 继承该 dtype
            local_sub_edge_index_list.append(local_edge_index.long())
        return local_sub_edge_index_list

    rowptr = structInfo.graph_csr_data['rowptr'].to(torch.long).cpu()
    col = structInfo.graph_csr_data['col'].to(torch.long).cpu()
    for partition in local_partitions:
        local_sub_edge_index_list.append(_subgraph_from_csr(partition, rowptr, col))
    return local_sub_edge_index_list


def _compute_local_spatial_pos(local_partitions, local_ppr_sub_edge_index_list, max_dist: int):
    spatial_pos_list = []
    for partition, local_edge_index in zip(local_partitions, local_ppr_sub_edge_index_list):
        num_nodes = int(partition.numel())
        dist_mat = torch.full((num_nodes, num_nodes), max_dist + 1, dtype=torch.long)
        if num_nodes == 0:
            spatial_pos_list.append(dist_mat)
            continue
        for i in range(num_nodes):
            dist_mat[i, i] = 0
        adjacency = [[] for _ in range(num_nodes)]
        if local_edge_index.numel() > 0:
            src = local_edge_index[0].tolist()
            dst = local_edge_index[1].tolist()
            for u, v in zip(src, dst):
                adjacency[u].append(v)
        for src in range(num_nodes):
            queue = deque([src])
            while queue:
                u = queue.popleft()
                current_dist = int(dist_mat[src, u].item())
                if current_dist >= max_dist:
                    continue
                for v in adjacency[u]:
                    if dist_mat[src, v] > current_dist + 1:
                        dist_mat[src, v] = current_dist + 1
                        queue.append(v)
        spatial_pos = torch.zeros_like(dist_mat)
        reachable = dist_mat <= max_dist
        spatial_pos[reachable] = dist_mat[reachable] + 1
        spatial_pos_list.append(spatial_pos)
    return spatial_pos_list


def _round_robin_assignment(num_windows: int, world_size: int) -> list[list[int]]:
    return [list(range(rank, num_windows, world_size)) for rank in range(world_size)]


def _edge_balanced_step_assignment(structInfo: StructInfo, world_size: int) -> list[list[int]]:
    wm = structInfo.wm
    num_windows = len(wm.partitioned_results)
    if num_windows == 0:
        return [[] for _ in range(world_size)]

    sub_edges = getattr(wm, 'sub_edge_index_for_partition_results', None) or []
    edge_counts = [
        int(sub_edges[pid].shape[1]) if pid < len(sub_edges) and sub_edges[pid] is not None else 0
        for pid in range(num_windows)
    ]
    node_counts = [int(wm.partitioned_results[pid].numel()) for pid in range(num_windows)]
    sorted_pids = sorted(
        range(num_windows),
        key=lambda pid: (edge_counts[pid], node_counts[pid], -pid),
        reverse=True,
    )

    assignments = [[] for _ in range(world_size)]
    rank_edge_totals = [0 for _ in range(world_size)]
    rank_node_totals = [0 for _ in range(world_size)]
    step_count = (num_windows + world_size - 1) // world_size

    for step in range(step_count):
        bucket = sorted_pids[step * world_size:(step + 1) * world_size]
        rank_order = sorted(
            range(world_size),
            key=lambda rank: (rank_edge_totals[rank], rank_node_totals[rank], len(assignments[rank]), rank),
        )
        for pid, rank in zip(bucket, rank_order):
            assignments[rank].append(pid)
            rank_edge_totals[rank] += edge_counts[pid]
            rank_node_totals[rank] += node_counts[pid]

    return assignments


def _build_window_assignment(args, structInfo: StructInfo) -> list[list[int]]:
    world_size = int(getattr(args, 'world_size', 1))
    num_windows = len(structInfo.wm.partitioned_results)
    if world_size <= 1:
        return [list(range(num_windows))]

    strategy = getattr(args, 'window_assignment_strategy', 'edge_balanced_step')
    if strategy == 'round_robin':
        return _round_robin_assignment(num_windows, world_size)
    if strategy == 'edge_balanced_step':
        return _edge_balanced_step_assignment(structInfo, world_size)
    raise ValueError(f"Unsupported window_assignment_strategy: {strategy}")


def _print_window_balance(args, structInfo: StructInfo, assignments: list[list[int]]):
    if getattr(args, 'rank', 0) != 0:
        return

    wm = structInfo.wm
    sub_edges = getattr(wm, 'sub_edge_index_for_partition_results', None) or []
    edge_counts = [
        int(sub_edges[pid].shape[1]) if pid < len(sub_edges) and sub_edges[pid] is not None else 0
        for pid in range(len(wm.partitioned_results))
    ]
    node_counts = [int(wm.partitioned_results[pid].numel()) for pid in range(len(wm.partitioned_results))]

    rank_edges = [sum(edge_counts[pid] for pid in rank_pids) for rank_pids in assignments]
    rank_nodes = [sum(node_counts[pid] for pid in rank_pids) for rank_pids in assignments]
    rank_windows = [len(rank_pids) for rank_pids in assignments]
    max_steps = max(rank_windows, default=0)
    step_imbalances = []
    for step in range(max_steps):
        step_edges = [
            edge_counts[rank_pids[step]]
            for rank_pids in assignments
            if step < len(rank_pids)
        ]
        if not step_edges:
            continue
        min_edges = min(step_edges)
        max_edges = max(step_edges)
        step_imbalances.append((max_edges / max(min_edges, 1)) if max_edges > 0 else 1.0)

    nonzero_rank_edges = [value for value in rank_edges if value > 0]
    rank_edge_imbalance = (
        max(nonzero_rank_edges) / max(min(nonzero_rank_edges), 1)
        if nonzero_rank_edges else 1.0
    )
    avg_step_imbalance = sum(step_imbalances) / len(step_imbalances) if step_imbalances else 1.0
    max_step_imbalance = max(step_imbalances) if step_imbalances else 1.0

    print(
        f"[WindowBalance] strategy={getattr(args, 'window_assignment_strategy', 'edge_balanced_step')} "
        f"windows={len(edge_counts)} steps={max_steps}"
    )
    print(f"[WindowBalance] rank_edges={rank_edges} rank_nodes={rank_nodes} rank_windows={rank_windows}")
    print(f"[WindowBalance] rank_edge_imbalance={rank_edge_imbalance:.6f}")
    print(f"[WindowBalance] step_edge_imbalance avg={avg_step_imbalance:.6f} max={max_step_imbalance:.6f}")


def build_local_partitions(structInfo: StructInfo, rank: int, world_size: int):
    return structInfo.local_partition_ids, structInfo.local_partitions


def build_dup_cache_metadata(structInfo: StructInfo, feature: torch.Tensor, device: str):
    local_dup_nodes = getattr(structInfo, 'local_dup_nodes_per_partition', None) or []
    if not local_dup_nodes:
        structInfo.local_dup_indices = []
        structInfo.local_dup_nodes_per_partition_feature = _empty_cache(feature, device)
        return torch.empty((0,), dtype=torch.long)

    non_empty_dup_nodes = [dup.to(torch.long).cpu() for dup in local_dup_nodes if dup.numel() > 0]
    if not non_empty_dup_nodes:
        structInfo.local_dup_indices = [torch.empty((0,), dtype=torch.long, device=device) for _ in local_dup_nodes]
        structInfo.local_dup_nodes_per_partition_feature = _empty_cache(feature, device)
        return torch.empty((0,), dtype=torch.long)

    dup_unique_sorted = torch.unique(torch.cat(non_empty_dup_nodes), sorted=True)
    structInfo.local_dup_indices = []
    for dup_nodes in local_dup_nodes:
        dup_nodes_cpu = dup_nodes.to(torch.long).cpu()
        if dup_nodes_cpu.numel() == 0:
            indices = torch.empty((0,), dtype=torch.long, device=device)
        else:
            indices = torch.searchsorted(dup_unique_sorted, dup_nodes_cpu).to(device=device, dtype=torch.long)
        structInfo.local_dup_indices.append(indices)

    structInfo.local_dup_nodes_per_partition_feature = feature[dup_unique_sorted].to(device)
    return dup_unique_sorted


def _build_local_bundle_for_rank(args, structInfo: StructInfo, rank: int, assignments=None):
    wm = structInfo.wm
    if assignments is None:
        assignments = _round_robin_assignment(len(wm.partitioned_results), int(getattr(args, 'world_size', 1)))
    local_partition_ids = assignments[rank]
    original_local_partitions = [wm.partitioned_results[pid].to(torch.long).cpu() for pid in local_partition_ids]
    local_partitions = original_local_partitions
    local_dup_nodes_per_partition = [torch.empty((0,), dtype=torch.long) for _ in local_partitions]

    cached_sub_edges = getattr(wm, 'sub_edge_index_for_partition_results', None) or []
    has_cached_sub_edges = len(cached_sub_edges) == len(wm.partitioned_results)
    local_sub_edge_index_list = None
    if has_cached_sub_edges:
        local_sub_edge_index_list = [cached_sub_edges[pid].to(torch.long).cpu() for pid in local_partition_ids]

    if args.use_cache:
        reranged_partitions, local_dup_nodes_per_partition = _compute_local_duplicate_nodes(local_partitions)
        if local_sub_edge_index_list is not None:
            local_sub_edge_index_list = [
                _remap_sub_edge_index_for_reordered_partition(old_part, new_part, old_edge_index)
                for old_part, new_part, old_edge_index in zip(
                    original_local_partitions,
                    reranged_partitions,
                    local_sub_edge_index_list,
                )
            ]
        local_partitions = reranged_partitions

    bundle = {
        'local_partition_ids': local_partition_ids,
        'local_partitions': local_partitions,
        'local_dup_nodes_per_partition': local_dup_nodes_per_partition,
    }
    if local_sub_edge_index_list is not None:
        bundle['local_sub_edge_index_list'] = local_sub_edge_index_list
    if args.struct_enc == 'True':
        bundle['local_ppr_sub_edge_index_list'] = _build_local_ppr_sub_edge_index_list(structInfo, local_partitions)
    return bundle


def _rebuild_local_window_structures(args, structInfo: StructInfo, feature: torch.Tensor, device: str, local_ppr_sub_edge_index_list):
    """
    根据rank所在的分区构建cache
    """
    timing_stats = {
        'local_dup_cache_rebuild_time': 0.0,
        'local_subgraph_rebuild_time': 0.0,
        'local_spatial_rebuild_time': 0.0,
    }

    dup_start = time.time()
    if args.use_cache:
        if len(structInfo.local_dup_nodes_per_partition) != len(structInfo.local_partitions):
            local_partitions, local_dup_nodes_per_partition = _compute_local_duplicate_nodes(structInfo.local_partitions)
            structInfo.local_partitions = local_partitions
            structInfo.local_dup_nodes_per_partition = local_dup_nodes_per_partition
        build_dup_cache_metadata(structInfo, feature, device)
    else:
        structInfo.local_dup_nodes_per_partition = [torch.empty((0,), dtype=torch.long) for _ in structInfo.local_partitions]
        structInfo.local_dup_indices = []
        structInfo.local_dup_nodes_per_partition_feature = _empty_cache(feature, device)
    timing_stats['local_dup_cache_rebuild_time'] = time.time() - dup_start

    subgraph_start = time.time()
    if len(structInfo.local_sub_edge_index_for_partition_results) == len(structInfo.local_partitions):
        structInfo.local_sub_edge_index_for_partition_results = [
            edge_index.to(torch.long).cpu()
            for edge_index in structInfo.local_sub_edge_index_for_partition_results
        ]
    else:
        structInfo.local_sub_edge_index_for_partition_results = _build_local_sub_edge_index_list(structInfo, structInfo.local_partitions)
    timing_stats['local_subgraph_rebuild_time'] = time.time() - subgraph_start

    spatial_start = time.time()
    if args.struct_enc == 'True':
        structInfo.local_spatial_pos_by_pid = _compute_local_spatial_pos(
            structInfo.local_partitions,
            local_ppr_sub_edge_index_list,
            max_dist=args.max_dist,
        )
    else:
        structInfo.local_spatial_pos_by_pid = []
    timing_stats['local_spatial_rebuild_time'] = time.time() - spatial_start
    return timing_stats


def broadcast_window_state(args, structInfo: StructInfo, feature: torch.Tensor, device: str):
    if args.rank == 0:
        print('[Preprocess] broadcast_window_state started: building window bundles...')
    timing_stats = {
        'bundle_write_time': 0.0,
        'bundle_load_time': 0.0,
        'local_dup_cache_rebuild_time': 0.0,
        'local_subgraph_rebuild_time': 0.0,
        'local_spatial_rebuild_time': 0.0,
        'window_state_total_time': 0.0,
    }
    overall_start = time.time()

    if args.rank == 0:
        # 把全局窗口结构保存在 CPU
        restore_global_window_state(structInfo)
        _stash_global_window_state_cpu(structInfo)

    if args.world_size <= 1:
        assignments = _build_window_assignment(args, structInfo)
        _print_window_balance(args, structInfo, assignments)
        bundle = _build_local_bundle_for_rank(args, structInfo, args.rank, assignments=assignments)
        _assign_local_window_bundle(structInfo, bundle)
        local_ppr_sub_edge_index_list = bundle.get('local_ppr_sub_edge_index_list', [])
        rebuild_stats = _rebuild_local_window_structures(args, structInfo, feature, device, local_ppr_sub_edge_index_list)
        timing_stats.update(rebuild_stats)
        if args.rank == 0:
            _release_hot_global_window_state(structInfo)
        timing_stats['window_state_total_time'] = time.time() - overall_start
        return timing_stats

    version = structInfo.window_state_version # 这是窗口状态版本号，避免 node_out/node_in 重新分发时覆盖混淆

    # 对每个 rank 生成一个 bundle, including
    # local_partition_ids、local_partitions、如果开结构编码，再加 local_ppr_sub_edge_index_list
    # 写到磁盘
    if args.rank == 0:
        bundle_write_start = time.time()
        assignments = _build_window_assignment(args, structInfo)
        _print_window_balance(args, structInfo, assignments)
        for rank in range(args.world_size):
            for path in (_bundle_path(args, version, rank), _done_path(args, version, rank)):
                if os.path.exists(path):
                    os.remove(path)
            bundle = _build_local_bundle_for_rank(args, structInfo, rank, assignments=assignments)
            _atomic_torch_save(bundle, _bundle_path(args, version, rank))
        timing_stats['bundle_write_time'] = time.time() - bundle_write_start

    # 每个 rank 读自己的 bundle，并本地重建结构
    _wait_for_path(_bundle_path(args, version, args.rank), timeout_seconds=_wait_timeout_seconds(args))
    bundle_load_start = time.time()
    bundle = torch.load(_bundle_path(args, version, args.rank), map_location='cpu')
    timing_stats['bundle_load_time'] = time.time() - bundle_load_start
    _assign_local_window_bundle(structInfo, bundle)     # 把本 rank 的 local 数据写进自己进程内的 structInfo，不会冲突
    local_ppr_sub_edge_index_list = bundle.get('local_ppr_sub_edge_index_list', [])
    rebuild_stats = _rebuild_local_window_structures(args, structInfo, feature, device, local_ppr_sub_edge_index_list)
    timing_stats.update(rebuild_stats)
    timing_stats['window_state_total_time'] = time.time() - overall_start
    print(f"[WindowState] rank={args.rank} timing={timing_stats}", flush=True)

    _touch_done(_done_path(args, version, args.rank), timing_stats)

    #  rank 0 删除这些临时 bundle 文件，并释放全局窗口对象
    if args.rank == 0:
        for rank in range(args.world_size):
            _wait_for_path(_done_path(args, version, rank), timeout_seconds=_wait_timeout_seconds(args))
        for rank in range(args.world_size):
            for path in (_bundle_path(args, version, rank), _done_path(args, version, rank)):
                if os.path.exists(path):
                    os.remove(path)
    if args.rank == 0:
        _release_hot_global_window_state(structInfo)
    return timing_stats
