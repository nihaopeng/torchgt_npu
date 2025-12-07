import os
import torch
import torch.nn.functional as F
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
    
    stats['mean'] = np.mean(off_diagonal)       # 均值
    stats['std'] = np.std(off_diagonal)         # 标准差
    stats['variance'] = np.var(off_diagonal)    # 方差
    stats['min'] = np.min(off_diagonal)         # 最小值
    stats['max'] = np.max(off_diagonal)         # 最大值
    stats['median'] = np.median(off_diagonal)   # 中位数
    
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
        # 对于高分注意力i->j，检查是否存在j->i的高分注意力
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
    分析高分注意力节点对之间的距离分布，计算高分注意力节点对邻居的占比
    
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
    
    # 添加高分点对中,不同距离的注意力分数数据
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

def analyze_node_coverage(attention_matrix: np.ndarray, threshold_percentile: float = 90, 
                          csv_path: str = "node_coverage_analysis.csv"):
    """
    分析节点的注意力覆盖范围：统计每个节点对"其他"节点产生高分注意力的数量和比例。
    并将统计结果保存到 CSV 文件。
    
    Args:
        attention_matrix: 注意力分数方阵 [N, N] (numpy array)
        threshold_percentile: 定义"高分"的分位数 (默认90, 即前10%的分数)
        csv_path: 结果保存的 CSV 文件路径
        
    Returns:
        dict: 包含覆盖数量和比例的统计信息
    """
    n = attention_matrix.shape[0]
    
    # 1. 准备数据：提取非对角线元素（关注"其他"顶点）
    mask_no_diag = ~np.eye(n, dtype=bool)
    off_diagonal_scores = attention_matrix[mask_no_diag]
    
    # 2. 确定"高分"阈值
    threshold = np.percentile(off_diagonal_scores, threshold_percentile)
    
    # 3. 逐行统计：每个节点关注了多少个高分邻居？
    attn_temp = attention_matrix.copy()
    np.fill_diagonal(attn_temp, -np.inf)
    
    # 统计每行中大于阈值的元素个数
    high_attn_counts = np.sum(attn_temp > threshold, axis=1)
    
    # 4. 计算比例
    ratios = high_attn_counts / (n - 1) if n > 1 else np.zeros_like(high_attn_counts, dtype=float)
    
    # 5. 稀疏性验证指标 (例如：关注 <= 50 个高分节点的节点比例)
    k_sparse = 50
    sparse_node_ratio = np.sum(high_attn_counts <= k_sparse) / n
    
    # 6. 打印统计报告
    print("=" * 60)
    print(f"节点注意力覆盖范围分析 (Top {100-threshold_percentile}%)")
    print("=" * 60)
    print(f"高分阈值: > {threshold:.6f}")
    print(f"[数量统计] 均值: {np.mean(high_attn_counts):.2f}, 中位数: {np.median(high_attn_counts):.0f}")
    print(f"[比例统计] 平均覆盖率: {np.mean(ratios):.2%}")
    print(f"[稀疏验证] 关注目标 <= {k_sparse} 的节点比例: {sparse_node_ratio:.2%}")
    
    # 7. 准备要保存的数据
    result_data = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'num_nodes': n,
        'threshold_percentile': threshold_percentile,
        'threshold_value': threshold,
        
        # 数量统计
        'count_mean': np.mean(high_attn_counts),
        'count_median': np.median(high_attn_counts),
        'count_std': np.std(high_attn_counts),
        'count_min': np.min(high_attn_counts),
        'count_max': np.max(high_attn_counts),
        
        # 比例统计
        'ratio_mean': np.mean(ratios),
        'ratio_median': np.median(ratios),
        'ratio_max': np.max(ratios),
        
        # 稀疏性指标
        'sparse_node_ratio_50': sparse_node_ratio  # 关注少于50个节点的比例
    }
    
    # 8. 保存到 CSV 文件
    df_result = pd.DataFrame([result_data])
    
    # 如果文件不存在，创建新文件并写入表头；如果存在，追加数据
    try:
        if not os.path.exists(csv_path):
            df_result.to_csv(csv_path, index=False)
        else:
            df_result.to_csv(csv_path, mode='a', header=False, index=False)
        print(f"统计结果已保存到: {csv_path}")
    except Exception as e:
        print(f"保存 CSV 失败: {e}")

    return {
        'mean_count': np.mean(high_attn_counts),
        'mean_ratio': np.mean(ratios),
        'counts': high_attn_counts 
    }
    
def analyze_incoming_attention(attention_matrix: np.ndarray, threshold_percentile: float = 90, 
                               csv_path: str = "incoming_attention_analysis.csv"):
    """
    统计是否存在某些顶点总是被其他顶点高分注意(核心节点)
    
    Args:
        attention_matrix: 注意力分数方阵 [N, N]
        threshold_percentile: 定义"高分"的分位数 (默认90)
        csv_path: 结果保存路径
        
    Returns:
        dict: 统计结果，包含核心节点的比例
    """
    n = attention_matrix.shape[0]
    
    # 1. 准备数据：提取非对角线元素（排除自注意力，只看"其他"顶点的关注）
    mask_no_diag = ~np.eye(n, dtype=bool)
    off_diagonal_scores = attention_matrix[mask_no_diag]
    
    # 2. 确定"高分"阈值
    threshold = np.percentile(off_diagonal_scores, threshold_percentile)
    
    # 3. 逐列统计：每个节点被多少个"其他"节点高分关注？ (axis=0 代表列方向，即作为目标节点)
    attn_temp = attention_matrix.copy()
    np.fill_diagonal(attn_temp, -np.inf) # 排除对角线干扰
    
    # 统计每列中被几个顶点高分关注
    incoming_counts = np.sum(attn_temp > threshold, axis=0)
    
    # 4. 定义"Hub节点" (总是被高分注意的节点)
    # 标准：被关注次数 > 平均值 
    mean_val = np.mean(incoming_counts)
    std_val = np.std(incoming_counts)
    hub_threshold = mean_val
    
    is_hub = incoming_counts > hub_threshold
    num_hubs = np.sum(is_hub)
    hub_ratio = num_hubs / n if n > 0 else 0
    
    # 5. 打印统计报告
    print("=" * 60)
    print(f"节点被关注度分析 (Attention Sinks Analysis)")
    print("=" * 60)
    print(f"高分阈值: > {threshold:.6f}")
    print(f"被关注次数统计 - 均值: {mean_val:.2f}, 最大值: {np.max(incoming_counts)}")
    print(f"Hub节点定义: 被关注次数 > {hub_threshold:.2f} (均值 + 2σ)")
    print(f"是否存在Hub节点? {'是' if num_hubs > 0 else '否'}")
    if num_hubs > 0:
        print(f"Hub节点数量: {num_hubs}")
        print(f"Hub节点比例: {hub_ratio:.2%}")
    
    # 6. 准备结果数据
    result_data = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'num_nodes': n,
        'threshold_percentile': threshold_percentile,
        
        # 基础统计
        'incoming_mean': mean_val,
        'incoming_std': std_val,
        'incoming_max': np.max(incoming_counts),
        'incoming_min': np.min(incoming_counts),
        
        # Hub 分析
        'hub_threshold_val': hub_threshold,
        'num_hubs': num_hubs,
        'hub_ratio': hub_ratio,
        
        # 极端情况：是否有人完全没人高分关注？
        'zero_high-attention_nodes': np.sum(incoming_counts == 0),
        'zero_high-attention_ratio': np.sum(incoming_counts == 0) / n
    }
    
    # 7. 保存到 CSV
    df_result = pd.DataFrame([result_data])
    try:
        if not os.path.exists(csv_path):
            df_result.to_csv(csv_path, index=False)
        else:
            df_result.to_csv(csv_path, mode='a', header=False, index=False)
        print(f"统计结果已保存到: {csv_path}")
    except Exception as e:
        print(f"保存 CSV 失败: {e}")
        
    return result_data   


def generate_mask_by_score(score, ratio=0.1, reserve_self_loop=True):
    """
    根据注意力分数动态生成掩码 (Dynamic Masking)。
    只保留全图中分数最高的前 ratio 比例的连接。
    
    Args:
        score: 注意力分数矩阵 [N, N] (Tensor, 已处理好维度)
        ratio: 保留比例 (float), 例如 0.1 代表保留 Top 10% 高分边。
        reserve_self_loop: 是否强制保留自环。
        
    Returns:
        mask: [1, 1, N, N] 的掩码张量，方便直接加到 attn_bias 上。
    """
    # 1. 确保是 Tensor
    if isinstance(score, np.ndarray):
        score = torch.from_numpy(score)
    
    device = torch.device('cpu')
    N = score.shape[0]
    
    # 2. 计算保留数量 (Per-Node K)
    k = int(N * ratio)
    k = max(1, min(k, N)) # 至少保留1个，最多N个
    
    # 3. 行内筛选 (Row-wise Top-K)
    # topk_indices: [N, k] -> 每一行保留的列索引
    _, topk_indices = torch.topk(score, k=k, dim=1)
    
    # 4. 构建掩码 (使用高级索引替代 scatter_)
    # 初始全为 -1e9 (断开)
    mask = torch.full((N, N), -1e9, device=device)
    
    # 构建行索引矩阵 [N, k]
    # 0, 0, ..., 0
    # 1, 1, ..., 1
    # ...
    row_indices = torch.arange(N, device=device).unsqueeze(1).expand(-1, k)
    
    # [核心操作] 高级索引赋值
    # mask[行坐标, 列坐标] = 0.0
    mask[row_indices, topk_indices] = 0.0
    
    # 5. 强制保留自环
    if reserve_self_loop:
        mask.fill_diagonal_(0.0)
        
    # 6. 维度适配 [1, 1, N, N]
    return mask.unsqueeze(0).unsqueeze(0)

def generate_mask_by_neighbor(score, edge_index, ratio=0.5, num_nodes=None,reserve_self_loop=True):
    """
    基于邻居的动态掩码 (Neighbor-based Dynamic Masking)。
    1. 找到它的所有物理邻居。
    2. 对这些邻居的 Attention Score 进行排序。
    3. 只保留排名前 (Degree_i * ratio) 的邻居，其余屏蔽。
    
    Args:
        score: 注意力分数 [N, N]
        edge_index: 物理边索引 [2, E]
        ratio: 邻居保留比例 (例如 0.5 表示只保留 50% 最好的邻居)
    """
    # 1. 准备数据
    if isinstance(score, np.ndarray):
        score = torch.from_numpy(score)

    device = torch.device('cpu')
    N = score.shape[0]
    
    # 2. 构建"仅邻居可见"的临时分数矩阵
    # 初始全为 -inf (非邻居)
    neighbor_only_score = torch.full((N, N), -float('inf'), device=device)
    
    # 只把物理邻居的分数填进去
    u, v = edge_index
    neighbor_only_score[u, v] = score[u, v]
    
    # 3. 计算每个节点的截断阈值 (Cutoff)
    # 统计每个节点的出度，bincount 统计每个节点 id 出现了几次，即有多少个邻居。
    degrees = torch.bincount(u, minlength=N)
    
    # 计算每个节点该保留多少个邻居，至少保留 1 个
    keep_counts = (degrees.float() * ratio).long().clamp(min=1)
    
    # 4. 行内排序 (Row-wise Sort)
    # 对 neighbor_only_score (仅邻居分数矩阵) 的每一行进行降序排列。
    # _: 排好序的分数 (我们不关心具体几分，只关心排名)。
    # sorted_indices: [N, N] 索引矩阵。
    #   第 i 行的内容是：节点 i 的邻居中，第 1 名是谁的索引，第 2 名是谁...
    _, sorted_indices = torch.sort(neighbor_only_score, dim=1, descending=True)
    
    # 5. 生成动态掩码

    # 扩展成 [1, N] 与 keep_counts [N, 1] 做比较
    col_grid = torch.arange(N, device=device).unsqueeze(0) # [1, N]
    # keep_counts 是 [5, 2, 1...] -> 扩展成 [N, 1]
    # 这代表每一行的保留的邻居
    cutoff_grid = keep_counts.unsqueeze(1)                 # [N, 1]
    
    # 核心逻辑：如果当前排位 < 该行应保留的数量，则为 True
    # 这是一个布尔矩阵 [N, N]，True 代表"该保留的排名位置"
    keep_rank_mask = col_grid < cutoff_grid
    
    # 6. 映射回原始坐标
    # 我们有了"排第几名该保留"，现在要把"第几名"翻译回"是哪个节点"
    # valid_rows: 对应的行号
    # valid_cols: 对应的列号 (即目标节点 ID)
    
    # 获取需要保留的位置的行号 (利用广播)
    rows_expanded = torch.arange(N, device=device).unsqueeze(1).expand(N, N)
    
    valid_rows = rows_expanded[keep_rank_mask]
    valid_cols = sorted_indices[keep_rank_mask]
    
    # 7. 构建最终掩码
    final_mask = torch.full((N, N), -1e9, device=device)
    final_mask[valid_rows, valid_cols] = 0.0
    
    # 8. 强制保留自环
    if reserve_self_loop:
        final_mask.fill_diagonal_(0.0)
        
    # 9. 维度适配 [1, 1, N, N]
    return final_mask.unsqueeze(0).unsqueeze(0)


# 下面这个先不弄
def analyze_distance_change(model, optimizer, x, y, edge_index, attn_bias, original_score: np.ndarray, 
                            steps: int = 500, percentile: float = 90, epoch: int = 0, 
                            save_dir: str = "distance_change_analysis"):
    """
    分析顶点距离改变后的注意力分数变化 
    Args:
        model: 模型 
        optimizer: 优化器
        x: 节点特征
        y: 标签
        edge_index: 边索引
        attn_bias: 注意力偏置
        original_score: 原始注意力分数矩阵 [N, N]
        steps: 微调步数 
        percentile: 筛选高分对的百分位阈值 (默认90)
        epoch: 当前训练轮数 (用于文件命名)
        save_dir: 结果保存目录
        
    Returns:
        list: 训练过程中的分数记录 (trace_records)
    """
    model.train()
    device = x.device
    N = x.shape[0]
    max_pairs = N // 2 
    
    # 1. 准备数据：筛选互不冲突的高分顶点对
    # 复制分数矩阵并排除对角线
    temp_score = original_score.copy()
    np.fill_diagonal(temp_score, -1) 
    
    valid_scores = temp_score[temp_score > -1]
    if len(valid_scores) == 0:
        return []
    
    # 确定阈值
    threshold_val = np.percentile(valid_scores, percentile)
    
    # 排序索引
    flat_indices = np.argsort(temp_score.ravel())[::-1]
    
    pairs = [] # 格式: (u, v, orig_score)
    visited = set()
    
    for idx in flat_indices:
        if len(pairs) >= max_pairs: 
            break
            
        u, v = np.unravel_index(idx, temp_score.shape)
        score_val = temp_score[u, v]
        
        if score_val < threshold_val: 
            break
        
        # 确保 u, v 未被占用且不是自环
        if u not in visited and v not in visited and u != v:
            pairs.append((u, v, score_val))
            visited.add(u)
            visited.add(v)
            
    if not pairs:
        print(f"[Analyze] 未找到满足阈值 ({threshold_val:.4f}) 的高分对")
        return []

    # 2. 打印实验开始报告
    print("=" * 60)
    print(f"距离变化分析 (Distance Change Experiment) - Epoch {epoch}")
    print("=" * 60)
    print(f"筛选阈值: Top {100-percentile}% (> {threshold_val:.4f})")
    print(f"选中对数: {len(pairs)}")
    print(f"原始均分: {np.mean([p[2] for p in pairs]):.4f}")

    # 3. 保存选中的顶点对信息到 CSV
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        df_pairs = pd.DataFrame([
            {"timestamp": timestamp, "epoch": epoch, "pair_id": i, "u": u, "v": v, "orig_score": s} 
            for i, (u, v, s) in enumerate(pairs)
        ])
        
        pairs_csv_path = os.path.join(save_dir, f"epoch_{epoch}_pairs.csv")
        df_pairs.to_csv(pairs_csv_path, index=False)
        print(f"顶点对信息已保存到: {pairs_csv_path}")

    # 4. 构建对抗序列 (两端拉伸)
    new_perm = np.full(N, -1, dtype=int)
    for i, (u, v, _) in enumerate(pairs):
        new_perm[i] = u           # 放在序列头部
        new_perm[N - 1 - i] = v   # 放在序列尾部
        
    # 填充剩余位置
    remaining = [n for n in range(N) if n not in visited]
    empty_slots = np.where(new_perm == -1)[0]
    fill_cnt = min(len(remaining), len(empty_slots))
    new_perm[empty_slots[:fill_cnt]] = remaining[:fill_cnt]
    
    perm_tensor = torch.tensor(new_perm, device=device)
    
    # 5. 数据重映射 
    old_to_new = torch.zeros(N, dtype=torch.long, device=device)
    old_to_new[perm_tensor] = torch.arange(N, device=device)
    
    x_perm = x[perm_tensor].clone()
    y_perm = y[perm_tensor].clone() if y is not None else None
    edge_index_perm = old_to_new[edge_index].clone()
    
    attn_bias_perm = None
    if attn_bias is not None:
        attn_bias_perm = attn_bias[perm_tensor][:, perm_tensor].clone()

    # 6. 微调训练并记录过程
    print(f"开始 {steps} 轮微调追踪...")
    trace_records = [] 
    
    for step in range(steps):
        optimizer.zero_grad()
        out, score_tensor = model(x_perm, attn_bias_perm, edge_index_perm, attn_type="full")
        
        if y_perm is not None:
            loss = F.nll_loss(out, y_perm)
            loss.backward()
            optimizer.step()
        
        # 提取当前分数
        with torch.no_grad():
            if score_tensor.dim() == 4: 
                s = score_tensor[0].mean(dim=0)
            elif score_tensor.dim() == 3: 
                s = score_tensor.mean(dim=0)
            else: 
                s = score_tensor
            
            # 记录每一对在当前 Step 的得分
            current_vals = []
            for i in range(len(pairs)):
                # u 在位置 i, v 在位置 N-1-i
                val = s[i, N - 1 - i].item()
                current_vals.append(val)
                
                trace_records.append({
                    "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "epoch": epoch,
                    "step": step + 1,
                    "pair_id": i,
                    "score": val
                })
            
            avg_curr = np.mean(current_vals)
            print(f"Step {step+1}: Loss={loss.item():.4f}, Avg Score={avg_curr:.4f}")

    # 7. 保存过程数据到 CSV
    if save_dir:
        df_trace = pd.DataFrame(trace_records)
        trace_csv_path = os.path.join(save_dir, f"epoch_{epoch}_trace.csv")
        df_trace.to_csv(trace_csv_path, index=False)
        print(f"过程记录已保存到: {trace_csv_path}")
        
    return trace_records