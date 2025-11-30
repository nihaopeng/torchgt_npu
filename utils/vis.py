import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']  # 或者 ['Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False

def draw_attn(attention_matrix: np.ndarray, edges: np.ndarray, partition_nodes, fig_path:str):
    # 创建子图节点ID到矩阵索引的映射
    node_id_to_index = {node_id: idx for idx, node_id in enumerate(partition_nodes.node_ids)}
    partition_node_ids_set = set(partition_nodes.node_ids)
    
    # 筛选出两个端点都在子图中的边，并映射到矩阵索引
    edges_list = []
    for e1, e2 in zip(edges[0], edges[1]):
        if e1 in partition_node_ids_set and e2 in partition_node_ids_set:
            idx1 = node_id_to_index[e1]
            idx2 = node_id_to_index[e2]
            # 确保索引在注意力矩阵范围内
            if idx1 < attention_matrix.shape[0] and idx2 < attention_matrix.shape[1]:
                edges_list.append([idx1, idx2])
    
    num_nodes = attention_matrix.shape[0]
    
    # 创建图
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(edges_list)
    
    # 提取边分数
    edge_scores = []
    valid_edges = []
    for u, v in G.edges():
        score = attention_matrix[u, v]
        edge_scores.append(score)
        valid_edges.append((u, v))
    
    if not edge_scores:
        print("没有有效的边可绘制")
        return
    
    # 选择高分边（前10%）
    threshold = np.percentile(edge_scores, 99)
    selected_edges = [(u, v) for (u, v), s in zip(valid_edges, edge_scores) if s >= threshold]
    
    if not selected_edges:
        print("没有边满足阈值条件")
        return
    
    selected_scores = [attention_matrix[u, v] for u, v in selected_edges]
    
    # 创建子图用于可视化
    H = nx.Graph()
    H.add_nodes_from(range(num_nodes))
    H.add_edges_from(selected_edges)
    
    # 归一化颜色
    selected_scores = np.array(selected_scores)
    if selected_scores.max() > selected_scores.min():
        edge_colors_norm = (selected_scores - selected_scores.min()) / (selected_scores.max() - selected_scores.min())
    else:
        edge_colors_norm = np.ones_like(selected_scores) * 0.5
    
    # 绘制图形
    pos = nx.spring_layout(H, seed=42)
    plt.figure(figsize=(10, 10))
    
    # 绘制节点
    nx.draw_networkx_nodes(H, pos, node_size=50, node_color='skyblue')
    
    # 绘制边
    edges_drawn = nx.draw_networkx_edges(
        H, pos, 
        edge_color=edge_colors_norm, 
        edge_cmap=plt.cm.Reds, 
        width=2,
        edge_vmin=0,
        edge_vmax=1
    )
    
    # 添加颜色条
    if edges_drawn is not None:
        plt.colorbar(edges_drawn, label="注意力强度")
    
    plt.title("子图注意力分数最高10%的边", fontsize=16)
    plt.axis("off")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"绘制完成: 节点数={num_nodes}, 边数={len(selected_edges)}")
