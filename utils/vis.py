import os
import torch
import numpy as np
import pandas as pd
from datetime import datetime

def calc_statistic(attention_matrix: torch.Tensor, fig_path: str):
    """
    统计注意力分数方阵的各种指标并可视化，结果累积保存到CSV文件
    
    参数:
    - attention_matrix: 注意力分数矩阵 (torch.Tensor)
    - fig_path: 图像保存路径
    """
    
    # 转换为numpy数组进行计算
    attn_np = attention_matrix
    n = attn_np.shape[0]
    
    # 1. 基础统计量
    stats = {}
    
    # 整体统计
    stats['matrix_size'] = n
    stats['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    stats['total_elements'] = n * n
    
    # 排除对角线（自注意力）的统计
    mask = ~np.eye(n, dtype=bool)
    off_diagonal = attn_np[mask]
    
    stats['mean'] = np.mean(off_diagonal)
    stats['std'] = np.std(off_diagonal)
    stats['variance'] = np.var(off_diagonal)
    stats['min'] = np.min(off_diagonal)
    stats['max'] = np.max(off_diagonal)
    stats['median'] = np.median(off_diagonal)
    
    # 分位数统计
    stats['q1'] = np.percentile(off_diagonal, 25)
    stats['q3'] = np.percentile(off_diagonal, 75)
    stats['iqr'] = stats['q3'] - stats['q1']
    
    # 高分注意力统计
    high_threshold_90 = np.percentile(off_diagonal, 90)
    high_threshold_95 = np.percentile(off_diagonal, 95)
    high_threshold_99 = np.percentile(off_diagonal, 99)
    
    stats['high_attention_90_percentile'] = high_threshold_90
    stats['high_attention_95_percentile'] = high_threshold_95
    stats['high_attention_99_percentile'] = high_threshold_99
    stats['high_attention_90_count'] = np.sum(off_diagonal >= high_threshold_90)
    stats['high_attention_95_count'] = np.sum(off_diagonal >= high_threshold_95)
    stats['high_attention_99_count'] = np.sum(off_diagonal >= high_threshold_99)
    
    # 2. 行级统计（每个顶点作为源顶点的统计）
    row_means = np.mean(attn_np, axis=1)
    row_stds = np.std(attn_np, axis=1)
    
    stats['row_mean_mean'] = np.mean(row_means)
    stats['row_mean_std'] = np.std(row_means)
    stats['row_std_mean'] = np.mean(row_stds)
    stats['row_std_std'] = np.std(row_stds)
    
    # 3. 列级统计（每个顶点作为目标顶点的统计）
    col_means = np.mean(attn_np, axis=0)
    col_stds = np.std(attn_np, axis=0)
    
    stats['col_mean_mean'] = np.mean(col_means)
    stats['col_mean_std'] = np.std(col_means)
    stats['col_std_mean'] = np.mean(col_stds)
    stats['col_std_std'] = np.std(col_stds)
    
    # 4. 注意力集中度统计
    # 使用基尼系数衡量注意力分布的不平等性
    def gini_coefficient(x):
        """计算基尼系数"""
        x = np.sort(x)
        n = len(x)
        index = np.arange(1, n + 1)
        return (np.sum((2 * index - n - 1) * x)) / (n * np.sum(x))
    
    row_gini = []
    for i in range(n):
        row = attn_np[i]
        if np.sum(row) > 0:
            row_gini.append(gini_coefficient(row))
    
    stats['gini_mean'] = np.mean(row_gini) if row_gini else 0
    stats['gini_std'] = np.std(row_gini) if row_gini else 0
    
    # 5. 稀疏性统计
    sparsity_threshold = stats['mean'] / 10  # 自定义稀疏阈值
    sparse_count = np.sum(off_diagonal < sparsity_threshold)
    stats['sparsity_ratio'] = sparse_count / len(off_diagonal)
    
    # 6. 保存统计结果到CSV
    csv_path = fig_path
    
    # 转换为DataFrame
    stats_df = pd.DataFrame([stats])
    
    # 如果CSV文件已存在，则追加；否则创建新文件
    if os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path)
        updated_df = pd.concat([existing_df, stats_df], ignore_index=True)
        updated_df.to_csv(csv_path, index=False)
    else:
        stats_df.to_csv(csv_path, index=False)
    
    # 7.4 高分注意力分析
    counts = [stats['high_attention_90_count'], stats['high_attention_95_count'], stats['high_attention_99_count']]
    percentages = [count / len(off_diagonal) * 100 for count in counts]
    
    # 8. 打印关键统计信息
    print("=" * 60)
    print("注意力矩阵统计分析结果")
    print("=" * 60)
    print(f"矩阵大小: {n}×{n}")
    print(f"均值: {stats['mean']:.6f} (±{stats['std']:.6f})")
    print(f"范围: [{stats['min']:.6f}, {stats['max']:.6f}]")
    print(f"中位数: {stats['median']:.6f}")
    print(f"稀疏度: {stats['sparsity_ratio']:.2%}")
    print(f"注意力不平等性 (基尼系数): {stats['gini_mean']:.4f}")
    print(f"高分注意力 (>90%分位): {stats['high_attention_90_count']} 个元素 ({percentages[0]:.2f}%)")
    print(f"统计结果已保存到: {csv_path}")
    print(f"可视化已保存到: {fig_path}")
    
    return stats

def edge_attn(attention_matrix: np.ndarray, partition_ids: np.ndarray, edges: np.ndarray, csv_path: str = "attention_analysis_results.csv"):
    """
    分析注意力分数分布，检查高分注意力是否集中在邻居节点上，并将结果保存到CSV文件
    
    Args:
        attention_matrix: 注意力分数方阵 [num_sub_node_ids, num_sub_node_ids]
        fig_path: 保留参数（为了兼容性）
        partition_ids: 子图节点在全局图中的ID [num_sub_node_ids]
        edges: 全局边关系 [2, num_edges]
        csv_path: CSV文件保存路径
    """
    num_nodes = len(partition_ids)
    
    # 1. 构建子图的邻接关系字典
    global_to_local = {global_id: local_id for local_id, global_id in enumerate(partition_ids)}
    
    # 构建邻接矩阵
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=bool)
    
    for i in range(edges.shape[1]):
        u, v = edges[0, i], edges[1, i]
        if u in global_to_local and v in global_to_local:
            local_u, local_v = global_to_local[u], global_to_local[v]
            adj_matrix[local_u, local_v] = True
            adj_matrix[local_v, local_u] = True  # 假设是无向图
    
    # 2. 收集注意力分数数据
    neighbor_attentions = []
    non_neighbor_attentions = []
    diagonal_attentions = []  # 自注意力
    
    for i in range(num_nodes):
        for j in range(num_nodes):
            attn_score = attention_matrix[i, j]
            if i == j:
                diagonal_attentions.append(attn_score)
            elif adj_matrix[i, j]:
                neighbor_attentions.append(attn_score)
            else:
                non_neighbor_attentions.append(attn_score)
    
    # 3. 统计分析
    neighbor_stats = {}
    non_neighbor_stats = {}
    diagonal_stats = {}
    
    if neighbor_attentions:
        neighbor_stats = {
            'mean': np.mean(neighbor_attentions),
            'max': np.max(neighbor_attentions),
            'median': np.median(neighbor_attentions),
            'std': np.std(neighbor_attentions),
            'count': len(neighbor_attentions)
        }
    
    if non_neighbor_attentions:
        non_neighbor_stats = {
            'mean': np.mean(non_neighbor_attentions),
            'max': np.max(non_neighbor_attentions),
            'median': np.median(non_neighbor_attentions),
            'std': np.std(non_neighbor_attentions),
            'count': len(non_neighbor_attentions)
        }
    
    if diagonal_attentions:
        diagonal_stats = {
            'mean': np.mean(diagonal_attentions),
            'max': np.max(diagonal_attentions),
            'median': np.median(diagonal_attentions),
            'std': np.std(diagonal_attentions),
            'count': len(diagonal_attentions)
        }
    
    # 4. 高分注意力分析
    threshold_90 = np.percentile(attention_matrix.flatten(), 90)  # 前10%作为高分注意力
    threshold_95 = np.percentile(attention_matrix.flatten(), 95)  # 前5%作为高分注意力
    
    high_attn_neighbor_90, high_attn_non_neighbor_90, high_attn_diagonal_90 = 0, 0, 0
    high_attn_neighbor_95, high_attn_non_neighbor_95, high_attn_diagonal_95 = 0, 0, 0
    
    for i in range(num_nodes):
        for j in range(num_nodes):
            attn_score = attention_matrix[i, j]
            if i == j:
                if attn_score > threshold_90:
                    high_attn_diagonal_90 += 1
                if attn_score > threshold_95:
                    high_attn_diagonal_95 += 1
            elif adj_matrix[i, j]:
                if attn_score > threshold_90:
                    high_attn_neighbor_90 += 1
                if attn_score > threshold_95:
                    high_attn_neighbor_95 += 1
            else:
                if attn_score > threshold_90:
                    high_attn_non_neighbor_90 += 1
                if attn_score > threshold_95:
                    high_attn_non_neighbor_95 += 1
    
    total_high_attn_90 = high_attn_neighbor_90 + high_attn_non_neighbor_90 + high_attn_diagonal_90
    total_high_attn_95 = high_attn_neighbor_95 + high_attn_non_neighbor_95 + high_attn_diagonal_95
    
    # 5. 每个节点的邻居注意力比例
    node_neighbor_ratios = []
    for i in range(num_nodes):
        neighbor_indices = np.where(adj_matrix[i])[0]
        if len(neighbor_indices) > 0:
            neighbor_attn = attention_matrix[i, neighbor_indices]
            total_attn = np.sum(attention_matrix[i]) - attention_matrix[i, i]  # 排除自注意力
            if total_attn > 0:
                ratio = np.sum(neighbor_attn) / total_attn
                node_neighbor_ratios.append(ratio)
    
    node_ratio_stats = {}
    if node_neighbor_ratios:
        node_ratio_stats = {
            'mean': np.mean(node_neighbor_ratios),
            'median': np.median(node_neighbor_ratios),
            'std': np.std(node_neighbor_ratios),
            'min': np.min(node_neighbor_ratios),
            'max': np.max(node_neighbor_ratios)
        }
    
    # 6. 准备结果数据
    result_data = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'num_nodes': num_nodes,
        'total_attention_pairs': num_nodes * num_nodes,
        
        # 邻居注意力统计
        'neighbor_attn_count': len(neighbor_attentions),
        'neighbor_attn_mean': neighbor_stats.get('mean', 0),
        'neighbor_attn_median': neighbor_stats.get('median', 0),
        'neighbor_attn_max': neighbor_stats.get('max', 0),
        'neighbor_attn_std': neighbor_stats.get('std', 0),
        
        # 非邻居注意力统计
        'non_neighbor_attn_count': len(non_neighbor_attentions),
        'non_neighbor_attn_mean': non_neighbor_stats.get('mean', 0),
        'non_neighbor_attn_median': non_neighbor_stats.get('median', 0),
        'non_neighbor_attn_max': non_neighbor_stats.get('max', 0),
        'non_neighbor_attn_std': non_neighbor_stats.get('std', 0),
        
        # 自注意力统计
        'diagonal_attn_count': len(diagonal_attentions),
        'diagonal_attn_mean': diagonal_stats.get('mean', 0),
        'diagonal_attn_median': diagonal_stats.get('median', 0),
        'diagonal_attn_max': diagonal_stats.get('max', 0),
        'diagonal_attn_std': diagonal_stats.get('std', 0),
        
        # 高分注意力分析 (90th percentile)
        'high_attn_threshold_90': threshold_90,
        'high_attn_total_90': total_high_attn_90,
        'high_attn_neighbor_90': high_attn_neighbor_90,
        'high_attn_neighbor_pct_90': high_attn_neighbor_90 / total_high_attn_90 * 100 if total_high_attn_90 > 0 else 0,
        'high_attn_non_neighbor_90': high_attn_non_neighbor_90,
        'high_attn_non_neighbor_pct_90': high_attn_non_neighbor_90 / total_high_attn_90 * 100 if total_high_attn_90 > 0 else 0,
        'high_attn_diagonal_90': high_attn_diagonal_90,
        'high_attn_diagonal_pct_90': high_attn_diagonal_90 / total_high_attn_90 * 100 if total_high_attn_90 > 0 else 0,
        
        # 高分注意力分析 (95th percentile)
        'high_attn_threshold_95': threshold_95,
        'high_attn_total_95': total_high_attn_95,
        'high_attn_neighbor_95': high_attn_neighbor_95,
        'high_attn_neighbor_pct_95': high_attn_neighbor_95 / total_high_attn_95 * 100 if total_high_attn_95 > 0 else 0,
        'high_attn_non_neighbor_95': high_attn_non_neighbor_95,
        'high_attn_non_neighbor_pct_95': high_attn_non_neighbor_95 / total_high_attn_95 * 100 if total_high_attn_95 > 0 else 0,
        'high_attn_diagonal_95': high_attn_diagonal_95,
        'high_attn_diagonal_pct_95': high_attn_diagonal_95 / total_high_attn_95 * 100 if total_high_attn_95 > 0 else 0,
        
        # 节点邻居注意力比例统计
        'node_neighbor_ratio_mean': node_ratio_stats.get('mean', 0),
        'node_neighbor_ratio_median': node_ratio_stats.get('median', 0),
        'node_neighbor_ratio_std': node_ratio_stats.get('std', 0),
        'node_neighbor_ratio_min': node_ratio_stats.get('min', 0),
        'node_neighbor_ratio_max': node_ratio_stats.get('max', 0),
    }
    
    # 7. 保存到CSV文件
    df_result = pd.DataFrame([result_data])
    
    # 如果文件不存在，创建新文件并写入表头；如果存在，追加数据
    if not os.path.exists(csv_path):
        df_result.to_csv(csv_path, index=False)
    else:
        df_result.to_csv(csv_path, mode='a', header=False, index=False)
    
    # 8. 打印简要结果
    print(f"分析完成! 结果已保存到: {csv_path}")
    print(f"节点数: {num_nodes}")
    print(f"邻居注意力比例 (前10%高分): {result_data['high_attn_neighbor_pct_90']:.1f}%")
    print(f"非邻居注意力比例 (前10%高分): {result_data['high_attn_non_neighbor_pct_90']:.1f}%")
    print(f"平均邻居注意力比例: {result_data['node_neighbor_ratio_mean']:.3f}")
    
    return result_data

import numpy as np
import pandas as pd
from collections import defaultdict, deque

def analyze_mutual_high_attention(attention_matrix: np.ndarray, partition_ids: np.ndarray, 
                                 edges: np.ndarray = None, threshold_percentile: float = 90,
                                 csv_path: str = "mutual_attention_analysis.csv"):
    """
    分析高分注意力的相互性，检查是否存在相互高分注意力
    
    Args:
        attention_matrix: 注意力分数方阵 [num_sub_node_ids, num_sub_node_ids]
        partition_ids: 子图节点在全局图中的ID [num_sub_node_ids]
        edges: 全局边关系 [2, num_edges] (可选)
        threshold_percentile: 用于定义高分注意力的百分位数阈值
        csv_path: CSV文件保存路径
    """
    num_nodes = len(partition_ids)
    
    # 1. 确定高分注意力阈值
    threshold = np.percentile(attention_matrix.flatten(), threshold_percentile)
    print(f"高分注意力阈值 ({threshold_percentile}%): {threshold:.4f}")
    
    # 2. 构建邻接矩阵（如果提供了边信息）
    adj_matrix = None
    if edges is not None:
        global_to_local = {global_id: local_id for local_id, global_id in enumerate(partition_ids)}
        adj_matrix = np.zeros((num_nodes, num_nodes), dtype=bool)
        
        for i in range(edges.shape[1]):
            u, v = edges[0, i], edges[1, i]
            if u in global_to_local and v in global_to_local:
                local_u, local_v = global_to_local[u], global_to_local[v]
                adj_matrix[local_u, local_v] = True
                adj_matrix[local_v, local_u] = True  # 假设是无向图
    
    # 3. 识别高分注意力对
    high_attention_pairs = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j and attention_matrix[i, j] > threshold:
                high_attention_pairs.append((i, j, attention_matrix[i, j]))
    
    # 4. 分析相互高分注意力
    # 创建高分注意力的字典
    high_attention_dict = defaultdict(list)
    for i, j, score in high_attention_pairs:
        high_attention_dict[i].append((j, score))
    
    mutual_pairs = []
    one_way_pairs = []
    
    # 检查相互性
    for i, j, score_ij in high_attention_pairs:
        # 检查是否存在j->i的高分注意力
        mutual = False
        score_ji = 0
        for k, score in high_attention_dict[j]:
            if k == i:
                mutual = True
                score_ji = score
                break
        
        if mutual:
            mutual_pairs.append({
                'node_i': i,
                'node_j': j,
                'global_id_i': partition_ids[i],
                'global_id_j': partition_ids[j],
                'score_ij': score_ij,
                'score_ji': score_ji,
                'avg_score': (score_ij + score_ji) / 2,
                'is_neighbor': adj_matrix[i, j] if adj_matrix is not None else None
            })
        else:
            one_way_pairs.append({
                'node_i': i,
                'node_j': j,
                'global_id_i': partition_ids[i],
                'global_id_j': partition_ids[j],
                'score': score_ij,
                'is_neighbor': adj_matrix[i, j] if adj_matrix is not None else None
            })
    
    # 5. 统计结果
    total_high_attention = len(high_attention_pairs)
    total_mutual = len(mutual_pairs)
    total_one_way = len(one_way_pairs)
    
    # 注意：相互对会被计算两次（i->j和j->i），所以实际相互对的数量是total_mutual/2
    actual_mutual_pairs = total_mutual // 2
    
    print("=== 相互高分注意力分析 ===")
    print(f"总高分注意力对数: {total_high_attention}")
    print(f"相互高分注意力对数: {actual_mutual_pairs}")
    print(f"单向高分注意力对数: {total_one_way}")
    print(f"相互高分注意力比例: {actual_mutual_pairs/(actual_mutual_pairs + total_one_way)*100:.2f}%")
    
    # 6. 分析相互高分注意力的特征
    mutual_scores_ij = [pair['score_ij'] for pair in mutual_pairs]
    mutual_scores_ji = [pair['score_ji'] for pair in mutual_pairs]
    one_way_scores = [pair['score'] for pair in one_way_pairs]
    
    # 7. 节点级别的相互性分析
    node_mutuality = defaultdict(lambda: {'out_high': 0, 'in_high': 0, 'mutual': 0})
    
    for i, j, score in high_attention_pairs:
        node_mutuality[i]['out_high'] += 1
    
    for pair in mutual_pairs:
        node_mutuality[pair['node_i']]['mutual'] += 1
        node_mutuality[pair['node_j']]['in_high'] += 1
    
    # 计算每个节点的相互性比例
    node_mutuality_ratios = []
    for node_id, counts in node_mutuality.items():
        if counts['out_high'] > 0:
            ratio = counts['mutual'] / counts['out_high']
            node_mutuality_ratios.append(ratio)
    
    if node_mutuality_ratios:
        print(f"\n=== 节点级别相互性分析 ===")
        print(f"平均相互性比例: {np.mean(node_mutuality_ratios):.4f}")
        print(f"相互性比例中位数: {np.median(node_mutuality_ratios):.4f}")
        print(f"相互性比例标准差: {np.std(node_mutuality_ratios):.4f}")
    
    # 8. 准备结果数据
    result_data = {
        'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'threshold_percentile': threshold_percentile,
        'threshold_value': threshold,
        'num_nodes': num_nodes,
        'total_high_attention_pairs': total_high_attention,
        'actual_mutual_pairs': actual_mutual_pairs,
        'one_way_pairs': total_one_way,
        'mutual_ratio': actual_mutual_pairs/(actual_mutual_pairs + total_one_way) if (actual_mutual_pairs + total_one_way) > 0 else 0,
        
        # 分数统计
        'mutual_score_ij_mean': np.mean(mutual_scores_ij) if mutual_scores_ij else 0,
        'mutual_score_ji_mean': np.mean(mutual_scores_ji) if mutual_scores_ji else 0,
        'mutual_score_avg_mean': np.mean([pair['avg_score'] for pair in mutual_pairs]) if mutual_pairs else 0,
        'one_way_score_mean': np.mean(one_way_scores) if one_way_scores else 0,
        
        # 节点级别统计
        'node_mutuality_ratio_mean': np.mean(node_mutuality_ratios) if node_mutuality_ratios else 0,
        'node_mutuality_ratio_median': np.median(node_mutuality_ratios) if node_mutuality_ratios else 0,
        'node_mutuality_ratio_std': np.std(node_mutuality_ratios) if node_mutuality_ratios else 0,
    }
    
    # 9. 保存到CSV文件
    df_result = pd.DataFrame([result_data])
    
    # 如果文件不存在，创建新文件并写入表头；如果存在，追加数据
    try:
        df_result.to_csv(csv_path, mode='a', header=False, index=False)
    except FileNotFoundError:
        df_result.to_csv(csv_path, index=False)
    
    print(f"\n结果已保存到: {csv_path}")
    
def analyze_attention_distance(attention_matrix: np.ndarray, partition_ids: np.ndarray, 
                              edges: np.ndarray, threshold_percentile: float = 90,
                              csv_path: str = "attention_distance_analysis.csv"):
    """
    分析高分注意力节点对之间的距离分布
    
    Args:
        attention_matrix: 注意力分数方阵 [num_sub_node_ids, num_sub_node_ids]
        partition_ids: 子图节点在全局图中的ID [num_sub_node_ids]
        edges: 全局边关系 [2, num_edges]
        threshold_percentile: 用于定义高分注意力的百分位数阈值
        csv_path: CSV文件保存路径
    """
    num_nodes = len(partition_ids)
    
    # 1. 确定高分注意力阈值
    threshold = np.percentile(attention_matrix.flatten(), threshold_percentile)
    print(f"高分注意力阈值 ({threshold_percentile}%): {threshold:.4f}")
    
    # 2. 构建全局图的邻接表
    global_to_local = {global_id: local_id for local_id, global_id in enumerate(partition_ids)}
    local_to_global = {local_id: global_id for local_id, global_id in enumerate(partition_ids)}
    
    # 构建全局图的邻接表（使用全局ID）
    global_adj_list = defaultdict(list)
    for i in range(edges.shape[1]):
        u, v = edges[0, i], edges[1, i]
        global_adj_list[u].append(v)
        global_adj_list[v].append(u)
    
    # 3. 计算所有节点对之间的最短路径距离
    print("计算节点间最短路径距离...")
    
    # 使用BFS计算最短路径
    def bfs_shortest_path(start_global_id):
        distances = {}
        visited = set()
        queue = deque([(start_global_id, 0)])
        
        while queue:
            node, dist = queue.popleft()
            if node in visited:
                continue
                
            visited.add(node)
            distances[node] = dist
            
            for neighbor in global_adj_list[node]:
                if neighbor not in visited:
                    queue.append((neighbor, dist + 1))
        
        return distances
    
    # 计算子图节点之间的距离矩阵
    distance_matrix = np.full((num_nodes, num_nodes), -1, dtype=int)
    
    for i in range(num_nodes):
        global_i = local_to_global[i]
        distances_from_i = bfs_shortest_path(global_i)
        
        for j in range(num_nodes):
            global_j = local_to_global[j]
            if global_j in distances_from_i:
                distance_matrix[i, j] = distances_from_i[global_j]
    
    # 4. 识别高分注意力节点对
    high_attention_pairs = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j and attention_matrix[i, j] > threshold:
                distance = distance_matrix[i, j]
                if distance != -1:  # 只考虑可达的节点对
                    high_attention_pairs.append({
                        'node_i': i,
                        'node_j': j,
                        'global_id_i': local_to_global[i],
                        'global_id_j': local_to_global[j],
                        'attention_score': attention_matrix[i, j],
                        'distance': distance,
                        'is_neighbor': (distance == 1)
                    })
    
    # 5. 分析高分注意力节点对的距离分布
    high_attention_distances = [pair['distance'] for pair in high_attention_pairs]
    
    if not high_attention_distances:
        print("未找到高分注意力节点对")
        return None
    
    # 计算距离统计
    distance_stats = {
        'mean': np.mean(high_attention_distances),
        'median': np.median(high_attention_distances),
        'std': np.std(high_attention_distances),
        'min': np.min(high_attention_distances),
        'max': np.max(high_attention_distances),
    }
    
    # 计算距离分布
    distance_counts = {}
    for dist in high_attention_distances:
        distance_counts[dist] = distance_counts.get(dist, 0) + 1
    
    # 计算邻居比例
    neighbor_pairs = sum(1 for pair in high_attention_pairs if pair['is_neighbor'])
    neighbor_ratio = neighbor_pairs / len(high_attention_pairs)
    
    print("=== 高分注意力距离分析 ===")
    print(f"高分注意力节点对总数: {len(high_attention_pairs)}")
    print(f"平均距离: {distance_stats['mean']:.2f}")
    print(f"距离中位数: {distance_stats['median']:.2f}")
    print(f"距离标准差: {distance_stats['std']:.2f}")
    print(f"最小距离: {distance_stats['min']}")
    print(f"最大距离: {distance_stats['max']}")
    print(f"邻居节点对比例: {neighbor_ratio*100:.2f}%")
    
    print(f"\n=== 距离分布 ===")
    for dist in sorted(distance_counts.keys()):
        count = distance_counts[dist]
        percentage = count / len(high_attention_pairs) * 100
        print(f"距离 {dist}: {count} 对 ({percentage:.2f}%)")
    
    # 6. 分析不同距离的注意力分数
    distance_attention_stats = {}
    for dist in set(high_attention_distances):
        scores = [pair['attention_score'] for pair in high_attention_pairs if pair['distance'] == dist]
        distance_attention_stats[dist] = {
            'count': len(scores),
            'mean_score': np.mean(scores),
            'median_score': np.median(scores),
            'max_score': np.max(scores) if scores else 0
        }
    
    print(f"\n=== 不同距离的注意力分数 ===")
    for dist in sorted(distance_attention_stats.keys()):
        stats = distance_attention_stats[dist]
        print(f"距离 {dist}: {stats['count']} 对, 平均分数: {stats['mean_score']:.4f}, "
              f"中位数: {stats['median_score']:.4f}, 最大值: {stats['max_score']:.4f}")
    
    # 7. 与随机注意力对的距离比较
    # 随机选择相同数量的节点对，计算平均距离
    num_random_pairs = len(high_attention_pairs)
    random_distances = []
    
    for _ in range(num_random_pairs):
        i, j = np.random.choice(num_nodes, 2, replace=False)
        distance = distance_matrix[i, j]
        if distance != -1:  # 只考虑可达的节点对
            random_distances.append(distance)
    
    if random_distances:
        random_mean = np.mean(random_distances)
        random_median = np.median(random_distances)
        
        print(f"\n=== 与随机节点对比较 ===")
        print(f"随机节点对平均距离: {random_mean:.2f}")
        print(f"随机节点对距离中位数: {random_median:.2f}")
        print(f"高分注意力平均距离与随机平均距离的比值: {distance_stats['mean']/random_mean:.4f}")
    
    # 8. 准备结果数据
    result_data = {
        'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'threshold_percentile': threshold_percentile,
        'threshold_value': threshold,
        'num_nodes': num_nodes,
        'num_high_attention_pairs': len(high_attention_pairs),
        
        # 距离统计
        'distance_mean': distance_stats['mean'],
        'distance_median': distance_stats['median'],
        'distance_std': distance_stats['std'],
        'distance_min': distance_stats['min'],
        'distance_max': distance_stats['max'],
        'neighbor_ratio': neighbor_ratio,
        
        # 距离分布
        'distance_1_count': distance_counts.get(1, 0),
        'distance_2_count': distance_counts.get(2, 0),
        'distance_3_count': distance_counts.get(3, 0),
        'distance_4_count': distance_counts.get(4, 0),
        'distance_5_plus_count': sum(count for dist, count in distance_counts.items() if dist >= 5),
        
        # 距离1的比例
        'distance_1_ratio': distance_counts.get(1, 0) / len(high_attention_pairs),
        'distance_2_ratio': distance_counts.get(2, 0) / len(high_attention_pairs),
        'distance_3_ratio': distance_counts.get(3, 0) / len(high_attention_pairs),
    }
    
    # 添加随机比较结果
    if random_distances:
        result_data.update({
            'random_distance_mean': random_mean,
            'random_distance_median': random_median,
            'distance_ratio_vs_random': distance_stats['mean'] / random_mean,
        })
    
    # 添加不同距离的注意力分数
    for dist in sorted(distance_attention_stats.keys()):
        if dist <= 5:  # 只保存前5个距离的统计
            stats = distance_attention_stats[dist]
            result_data[f'distance_{dist}_mean_score'] = stats['mean_score']
            result_data[f'distance_{dist}_median_score'] = stats['median_score']
    
    # 9. 保存详细的高分注意力对信息到CSV
    if high_attention_pairs:
        pairs_df = pd.DataFrame(high_attention_pairs)
        pairs_df.to_csv(csv_path.replace('.csv', '_pairs.csv'), index=False)
        print(f"\n高分注意力对详细信息已保存到: {csv_path.replace('.csv', '_pairs.csv')}")
    
    # 10. 保存汇总结果到CSV
    df_result = pd.DataFrame([result_data])
    
    # 如果文件不存在，创建新文件并写入表头；如果存在，追加数据
    try:
        existing_df = pd.read_csv(csv_path)
        df_result.to_csv(csv_path, mode='a', header=False, index=False)
    except FileNotFoundError:
        df_result.to_csv(csv_path, index=False)
    
    print(f"汇总结果已保存到: {csv_path}")
    
    # 11. 返回详细信息用于进一步分析
    detailed_results = {
        'high_attention_pairs': high_attention_pairs,
        'distance_stats': distance_stats,
        'distance_counts': distance_counts,
        'distance_attention_stats': distance_attention_stats,
        'result_data': result_data
    }
    
    if random_distances:
        detailed_results['random_distance_stats'] = {
            'mean': random_mean,
            'median': random_median
        }
    
    return detailed_results

