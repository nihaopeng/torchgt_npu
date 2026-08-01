#!/usr/bin/env python3
"""
================================================================================
动机实验：验证图分区方案对全注意力高分顶点对的捕获能力
================================================================================

实验设计
--------
1. 全注意力训练至收敛 → 提取最后一层的 post-softmax 注意力权重矩阵 [N, N]
2. 从注意力矩阵中筛选高分顶点对（top-k%）
3. 计算分区方案的"捕获率" = 全注意力高分对中，两端点出现在同一分区内的比例

两种捕获率：
  - 结构捕获率（structural）: 仅看分区结构本身是否把高分对放入同一窗口，无需训练分区模型
  - 分区训练捕获率（trained）  : 在分区方案上训练模型至收敛后，再次检查捕获率

输出
----
  - results/ 目录下的对比图表
  - 控制台打印各项指标

使用方法
--------
  python scripts/motivation_capture_rate.py --dataset cora --mode both
  python scripts/motivation_capture_rate.py --dataset amazon --mode structural --subgraph_nodes 3000
  python scripts/motivation_capture_rate.py --dataset ogbn-arxiv --mode both --expand

依赖
----
  - PyTorch, pymetis (分区), matplotlib (绘图)
  - 项目内模块：models/graphormer_dist_node_level, core/ppr_preprocess
  - 数据集：从 NeutronGT/dataset/<name>/ 下的 x.pt, y.pt, edge_index.pt 加载
================================================================================
"""

import argparse
import os
import sys
import time
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")  # 无头环境兼容
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 配置中文字体，避免绘图乱码
_font_paths = fm.findSystemFonts(fontpaths=["/usr/share/fonts/opentype/noto/"])
for _fp in _font_paths:
    if "NotoSansCJK" in _fp:
        fm.fontManager.addfont(_fp)
        _prop = fm.FontProperties(fname=_fp)
        _font_name = _prop.get_name()
        plt.rcParams["font.sans-serif"] = [_font_name, "DejaVu Sans"]
        break
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# ---------- 将项目根目录加入 path，以便导入项目内模块 ----------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ============================================================================
# 第 1 部分：参数解析
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Motivation: partition capture rate experiment")
    parser.add_argument("--dataset", type=str, default="cora",
                        choices=["cora", "citeseer", "pubmed",
                                 "ogbn-arxiv", "reddit", "ogbn-products",
                                 "amazon", "papers100"],
                        help="数据集名称。大图（除 cora/citeseer/pubmed 外）会自动子图采样")
    parser.add_argument("--dataset_dir", type=str,
                        default=os.path.join(_PROJECT_ROOT, "dataset"),
                        help="数据集根目录（包含各数据集子目录）")
    parser.add_argument("--mode", type=str, default="both",
                        choices=["structural", "trained", "both"],
                        help="实验模式: structural=仅结构捕获率, "
                             "trained=仅分区训练捕获率, both=两者都跑")
    parser.add_argument("--n_parts", type=int, default=4,
                        help="分区数量")
    parser.add_argument("--expand", action="store_true", default=False,
                        help="启用分区扩展（与训练流程一致的增广策略）")
    parser.add_argument("--aug_strategy", type=str, default="ours",
                        choices=["ours", "hub", "related", "random"],
                        help="分区增广策略：ours=related+hub+随机填充, hub=高度节点, "
                             "related=边界邻居, random=随机（仅 --expand 生效）")
    parser.add_argument("--window_related_ratio", type=float, default=0.15,
                        help="边界邻居引入比例（相对于核心分区大小，默认 0.15）")
    parser.add_argument("--window_hub_ratio", type=float, default=0.15,
                        help="高度 hub 节点引入比例（相对于核心分区大小，默认 0.15）")
    parser.add_argument("--window_random_ratio", type=float, default=0.0,
                        help="随机填充节点比例（相对于核心分区大小，默认 0.0）")
    parser.add_argument("--subgraph_nodes", type=int, default=3000,
                        help="大图采样子图的目标节点数（默认 3000）")
    parser.add_argument("--sample_hops", type=int, default=2,
                        help="子图采样的跳数（默认 2）")
    parser.add_argument("--topk", type=int, default=64,
                        help="Top-k 值")
    parser.add_argument("--hidden_dim", type=int, default=128,
                        help="隐藏层维度")
    parser.add_argument("--n_layers", type=int, default=3,
                        help="Transformer 层数")
    parser.add_argument("--num_heads", type=int, default=4,
                        help="注意力头数")
    parser.add_argument("--ffn_dim", type=int, default=256,
                        help="FFN 维度")
    parser.add_argument("--dropout_rate", type=float, default=0.1,
                        help="Dropout 率")
    parser.add_argument("--attention_dropout_rate", type=float, default=0.1,
                        help="注意力 dropout 率")
    parser.add_argument("--epochs_full", type=int, default=300,
                        help="全注意力模型训练轮数")
    parser.add_argument("--epochs_partition", type=int, default=300,
                        help="分区模型训练轮数")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="学习率")
    parser.add_argument("--weight_decay", type=float, default=5e-4,
                        help="权重衰减")
    parser.add_argument("--topk_ratios", type=str, default="0.01,0.02,0.05,0.10,0.20",
                        help="逗号分隔的 top-k 比例列表，用于计算捕获率曲线")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--device", type=str, default="cuda",
                        help="计算设备: cuda / cpu")
    parser.add_argument("--results_dir", type=str, default="./results/motivation",
                        help="图表输出目录")
    return parser.parse_args()


# ============================================================================
# 第 2 部分：数据加载
# ============================================================================

# 需要子图采样的大数据集
_LARGE_DATASETS = {"ogbn-arxiv", "reddit", "ogbn-products", "amazon", "papers100"}


def load_dataset(name: str, dataset_dir: str = "./dataset"):
    """
    加载数据集。cora/citeseer/pubmed 读本地 .pt，大图从 PyG/OGB 下载。

    返回: feature [N, D], y [N], edge_index [2, E], num_classes
    """
    # 小数据集：读缓存 .pt
    if name not in _LARGE_DATASETS:
        data_path = os.path.join(dataset_dir, name)
        x_path = os.path.join(data_path, "x.pt")
        y_path = os.path.join(data_path, "y.pt")
        ei_path = os.path.join(data_path, "edge_index.pt")
        if not all(os.path.exists(p) for p in [x_path, y_path, ei_path]):
            raise FileNotFoundError(
                f"{name} 的 .pt 文件缺失，请确保 {data_path}/ 下有 x.pt, y.pt, edge_index.pt。"
            )
        feature = torch.load(x_path, map_location="cpu")
        y = torch.load(y_path, map_location="cpu")
        edge_index = torch.load(ei_path, map_location="cpu")
        num_classes = int(y.max().item()) + 1
        print(f"[数据] {name}: 节点数={feature.shape[0]}, 特征维度={feature.shape[1]}, "
              f"边数={edge_index.shape[1]}, 类别数={num_classes}")
        return feature, y, edge_index, num_classes

    # 大数据集：优先缓存 .pt，没有则从 PyG / OGB 下载
    data_path = os.path.join(dataset_dir, name)
    os.makedirs(data_path, exist_ok=True)
    x_path = os.path.join(data_path, "x.pt")
    y_path = os.path.join(data_path, "y.pt")
    ei_path = os.path.join(data_path, "edge_index.pt")

    # papers100M 全量过大不缓存，其余大数据集缓存 .pt
    _skip_cache = (name == "papers100")

    if not _skip_cache and all(os.path.exists(p) for p in [x_path, y_path, ei_path]):
        print(f"[数据] 从缓存加载 {name} ...")
        feature = torch.load(x_path, map_location="cpu")
        y = torch.load(y_path, map_location="cpu")
        edge_index = torch.load(ei_path, map_location="cpu")
    else:
        hint = "从 OGB/PyG 加载（本地缓存）" if _skip_cache else "从源下载"
        print(f"[数据] {hint} {name} ...")

        if name == "reddit":
            from torch_geometric.datasets import Reddit
            data = Reddit(root=data_path)[0]
            feature, y, edge_index = data.x.float(), data.y.squeeze().long(), data.edge_index.long()

        elif name == "amazon":
            from torch_geometric.datasets import Amazon
            data = Amazon(root=data_path, name="Computers")[0]
            feature, y, edge_index = data.x.float(), data.y.squeeze().long(), data.edge_index.long()

        elif name in ("ogbn-arxiv", "ogbn-products", "papers100"):
            from ogb.nodeproppred import NodePropPredDataset
            ogb_name = {
                "ogbn-arxiv": "ogbn-arxiv",
                "ogbn-products": "ogbn-products",
                "papers100": "ogbn-papers100M",
            }[name]
            # OGB 在 root 下自动创建 {name}/ 子目录，所以 root 用 dataset_dir
            dataset = NodePropPredDataset(name=ogb_name, root=dataset_dir)
            graph, label = dataset[0]
            feature = torch.tensor(graph["node_feat"], dtype=torch.float32)
            y = torch.tensor(label, dtype=torch.long).squeeze()
            edge_index = torch.tensor(graph["edge_index"], dtype=torch.long)

        else:
            raise ValueError(f"未知数据集: {name}")

        # 缓存全量数据
        if not _skip_cache:
            torch.save(feature, x_path)
            torch.save(y, y_path)
            torch.save(edge_index, ei_path)
            print(f"[数据] 已缓存到 {data_path}/")

    num_classes = int(y.max().item()) + 1
    print(f"[数据] {name}: 节点数={feature.shape[0]}, 特征维度={feature.shape[1]}, "
          f"边数={edge_index.shape[1]}, 类别数={num_classes}")
    return feature, y, edge_index, num_classes


def random_split_idx(y, frac_train=0.6, frac_valid=0.2, frac_test=0.2, seed=42):
    """随机划分训练/验证/测试集。与项目中 gt_sp/utils.py 的逻辑保持一致。"""
    N = y.shape[0]
    indices = torch.randperm(N, generator=torch.Generator().manual_seed(seed))
    n_train = int(N * frac_train)
    n_valid = int(N * frac_valid)
    return {
        "train": indices[:n_train],
        "valid": indices[n_train:n_train + n_valid],
        "test": indices[n_train + n_valid:],
    }


# ============================================================================
# 第 2.5 部分：大图子图采样
# ============================================================================

def _build_csr(edge_index, N):
    """从 edge_index [2, E] 构建 CSR 格式 (indptr, indices)，使用 numpy 避免 Python 内存膨胀。"""
    import numpy as _np
    ei = edge_index.numpy()
    E = ei.shape[1]
    # 无向化：将反向边也加入
    all_src = _np.concatenate([ei[0], ei[1]])
    all_dst = _np.concatenate([ei[1], ei[0]])
    # 按 src 排序
    order = _np.argsort(all_src, kind='stable')
    all_src = all_src[order]
    all_dst = all_dst[order]
    # 构建 indptr
    indptr = _np.zeros(N + 1, dtype=_np.int64)
    _np.add.at(indptr, all_src + 1, 1)
    _np.cumsum(indptr, out=indptr)
    indices = all_dst.astype(_np.int64)
    return indptr, indices


def sample_subgraph(feature, y, edge_index, target_nodes=3000, n_hops=2, seed=42):
    """
    对大图进行邻居扩展采样子图，使其能跑 O(N²) 全注意力。

    小图（<1M 节点）：建 Python 邻接表 BFS。
    超大图（≥1M 节点）：用 CSR 格式（~14GB on papers100）BFS，避免 OOM。

    Returns:
        sub_feature [n_sub, D], sub_y [n_sub], sub_edge_index [2, E_sub], num_classes
    """
    import random as _random
    import numpy as _np
    torch.manual_seed(seed)
    N = feature.shape[0]
    target = min(target_nodes, N)
    edge_index_cpu = edge_index.cpu()
    _USE_CSR = (N >= 1_000_000)

    if _USE_CSR:
        print(f"[采样] 构建 CSR 格式（{N} 节点, {edge_index_cpu.shape[1]} 边）...")
        indptr, indices = _build_csr(edge_index_cpu, N)
    else:
        # 小图：直接建 Python 邻接表
        adj = [[] for _ in range(N)]
        for u, v in edge_index_cpu.t().tolist():
            adj[u].append(v)
            adj[v].append(u)

    # BFS 种子扩展
    visited = set()
    seeds = torch.randperm(N)[:max(50, target // 100)].tolist()
    frontier = list(seeds)
    visited.update(seeds)

    for _ in range(n_hops):
        next_frontier = []
        for node in frontier:
            if _USE_CSR:
                start, end = int(indptr[node]), int(indptr[node + 1])
                deg = end - start
                if deg == 0:
                    continue
                neighbors = indices[start:end].tolist()
                if deg > 100:
                    neighbors = _random.sample(neighbors, 100)
            else:
                neighbors = adj[node]
                if len(neighbors) > 100:
                    neighbors = _random.sample(neighbors, 100)
            for nb in neighbors:
                if nb not in visited:
                    visited.add(nb)
                    next_frontier.append(nb)
                    if len(visited) >= target * 2:
                        break
            if len(visited) >= target * 2:
                break
        frontier = next_frontier
        if len(visited) >= target * 2:
            break

    # 若超过 target*2，随机截取
    node_list = list(visited)
    if len(node_list) > target * 2:
        node_list = _random.sample(node_list, target * 2)
    elif len(node_list) < 100:
        remaining = list(set(range(N)) - set(node_list))
        extra = _random.sample(remaining, min(100 - len(node_list), len(remaining)))
        node_list.extend(extra)

    node_list.sort()
    n_sub = len(node_list)
    old_to_new = {old: i for i, old in enumerate(node_list)}
    node_set = set(node_list)

    # 分块提取子图边
    E = edge_index_cpu.shape[1]
    chunk_size = 50_000_000
    in_selected = torch.zeros(N, dtype=torch.bool)
    in_selected[torch.tensor(node_list)] = True

    sub_edge_parts = []
    for start in range(0, E, chunk_size):
        end = min(start + chunk_size, E)
        chunk = edge_index_cpu[:, start:end]
        mask = in_selected[chunk[0]] & in_selected[chunk[1]]
        if mask.any():
            sub_edge_parts.append(chunk[:, mask])

    if sub_edge_parts:
        sub_edge_full = torch.cat(sub_edge_parts, dim=1)
        sub_edges = []
        for u, v in sub_edge_full.t().tolist():
            sub_edges.append((old_to_new[u], old_to_new[v]))
        sub_edge_index = torch.tensor(sub_edges, dtype=torch.long).t()
    else:
        sub_edge_index = torch.empty((2, 0), dtype=torch.long)

    sub_feature = feature[node_list].clone()
    sub_y = y[node_list].clone()
    unique_labels = torch.unique(sub_y)
    label_map = {old.item(): new for new, old in enumerate(unique_labels)}
    sub_y = torch.tensor([label_map[l.item()] for l in sub_y], dtype=torch.long)
    num_classes = len(unique_labels)

    print(f"[采样] {n_sub} 节点, {sub_edge_index.shape[1]} 边, {num_classes} 类 "
          f"(原始: {N} 节点, {E} 边)")

    return sub_feature, sub_y, sub_edge_index, num_classes


# ============================================================================
# 第 3 部分：模型构建（使用项目中的 Graphormer，无需分布式初始化）
# ============================================================================

def build_full_attention_model(feature_dim, num_classes, args):
    """
    构建 Graphormer 模型用于全图全注意力训练。
    Graphormer 的 MultiHeadAttention 不含 dist_attn，因此无需初始化分布式环境。
    struct_enc 设为 False，跳过结构编码以简化实验。
    """
    from models.graphormer_dist_node_level import Graphormer

    # 构建一个 SimpleNamespace 模拟训练参数，供模型内部 (如 self.args.struct_enc) 使用
    model_args = SimpleNamespace()
    model_args.struct_enc = "False"

    model = Graphormer(
        n_layers=args.n_layers,
        num_heads=args.num_heads,
        input_dim=feature_dim,
        hidden_dim=args.hidden_dim,
        output_dim=num_classes,
        attn_bias_dim=args.num_heads,          # attn_bias 的 head 维度
        dropout_rate=args.dropout_rate,
        input_dropout_rate=args.dropout_rate,
        attention_dropout_rate=args.attention_dropout_rate,
        ffn_dim=args.ffn_dim,
        num_global_node=0,                      # 无全局 token
        args=model_args,
        num_in_degree=512,
        num_out_degree=512,
        num_spatial=32,
        num_edges=1024,
        max_dist=5,
        edge_dim=64,
    )
    return model


# ============================================================================
# 第 4 部分：训练与评估
# ============================================================================

def train_one_epoch(model, feature, y, edge_index, split_idx, optimizer, device, attn_type="full"):
    """全图单 epoch 训练。返回平均 loss。"""
    model.train()
    x = feature.to(device)
    target = y.to(device)
    edge_idx_gpu = edge_index.to(device) if edge_index is not None else None

    optimizer.zero_grad()
    # Graphormer.forward 参数说明:
    #   x, attn_bias, edge_index, in_degree, out_degree, spatial_pos, edge_input,
    #   perturb, attn_type, mask, pruning_mask, dup_nodes_kv_cache, part_id
    out, _, score_spe, _ = model(
        x, attn_bias=None, edge_index=edge_idx_gpu,
        in_degree=None, out_degree=None, spatial_pos=None, edge_input=None,
        attn_type=attn_type,
    )
    # out 已经是 F.log_softmax 的结果 → 直接使用 NLL loss
    train_mask = split_idx["train"].to(device)
    loss = F.nll_loss(out[train_mask], target[train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, feature, y, edge_index, split_idx, device, attn_type="full"):
    """全图评估。返回 (train_acc, valid_acc, test_acc)。"""
    model.eval()
    x = feature.to(device)
    target = y.to(device)
    edge_idx_gpu = edge_index.to(device) if edge_index is not None else None

    out, _, _, _ = model(
        x, attn_bias=None, edge_index=edge_idx_gpu,
        in_degree=None, out_degree=None, spatial_pos=None, edge_input=None,
        attn_type=attn_type,
    )
    pred = out.argmax(dim=1)

    results = {}
    for key in ["train", "valid", "test"]:
        mask = split_idx[key].to(device)
        acc = (pred[mask] == target[mask]).float().mean().item()
        results[key] = acc
    return results["train"], results["valid"], results["test"]


def train_full_epochs(model, feature, y, edge_index, split_idx, device,
                       attn_type, epochs, lr, weight_decay):
    """全注意力训练，跑满所有 epoch，不做早停。"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_history = []
    best_val = -1.0

    for epoch in range(epochs):
        loss = train_one_epoch(model, feature, y, edge_index, split_idx, optimizer, device, attn_type)
        loss_history.append(loss)
        train_acc, val_acc, test_acc = evaluate(model, feature, y, edge_index, split_idx, device, attn_type)
        if val_acc > best_val:
            best_val = val_acc

        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}: loss={loss:.4f}, "
                  f"train_acc={train_acc:.4f}, val_acc={val_acc:.4f}, test_acc={test_acc:.4f}")

    return model, loss_history, best_val


# ============================================================================
# 第 5 部分：提取注意力权重矩阵
# ============================================================================

@torch.no_grad()
def extract_pair_scores(model, feature, edge_index, device):
    """
    跑一次全图前向，从各层 CoreAttention._pair_scores 中提取对级注意力权重。
    返回最后一层的 [N, N] 矩阵（已取上三角化对称）。

    _pair_scores 由 full_attention() 在 softmax 后自动保存，shape = [N, N]。
    """
    model.eval()
    x = feature.to(device)
    edge_idx_gpu = edge_index.to(device) if edge_index is not None else None
    model(x, attn_bias=None, edge_index=edge_idx_gpu,
          in_degree=None, out_degree=None, spatial_pos=None, edge_input=None,
          attn_type="full")

    # 遍历所有 EncoderLayer → MultiHeadAttention → CoreAttention，收集 _pair_scores
    pair_scores_by_layer = []
    for layer in model.layers:                              # EncoderLayer
        mha = layer.self_attention                          # MultiHeadAttention
        local_attn = mha.local_attn                         # CoreAttention
        if hasattr(local_attn, "_pair_scores") and local_attn._pair_scores is not None:
            pair_scores_by_layer.append(local_attn._pair_scores.clone())

    if not pair_scores_by_layer:
        raise RuntimeError("未找到 _pair_scores！请确认 full_attention 中已添加 _pair_scores 赋值。")

    # 使用最后一层，并对称化: A_sym = (A + A^T) / 2
    A = pair_scores_by_layer[-1]  # [N, N]
    A_sym = (A + A.T) / 2.0
    return A_sym


# ============================================================================
# 第 6 部分：分区构建
# ============================================================================

def build_partitions(edge_index, feature, n_parts, n_nodes, topk, expand=False,
                     aug_strategy="ours",
                     window_related_ratio=0.15, window_hub_ratio=0.15,
                     window_random_ratio=0.0, seed=42):
    """
    构建分区。

    基础模式（expand=False）：PPR + 单层 Metis。
    扩展模式（expand=True）：与训练流程一致，包含：
      1. PPR + 两层父子 Metis
      2. 分区增广：边界邻居（related）+ 高度 hub 节点 + 随机填充

    返回 partitioned_nodes: list[Tensor]，每个元素是一个分区内的全局节点索引。
    """
    from core.ppr_preprocess import personal_pagerank, build_adj_fromat

    print(f"[分区] 计算 PPR (topk={topk})...")
    sorted_ppr = personal_pagerank(edge_index, alpha=0.85, topk=topk)

    if not expand:
        from core.ppr_preprocess import metis_partition
        print(f"[分区] 构建 CSR 邻接...")
        csr_adj, eweights, _ = build_adj_fromat(sorted_ppr)
        print(f"[分区] Metis 划分为 {n_parts} 个分区...")
        partitions = metis_partition(csr_adj, eweights, n_parts)
    else:
        from core.metisPartition import weightMetis_keepParent
        print(f"[分区] 构建 CSR 邻接...")
        csr_adj, eweights, _ = build_adj_fromat(sorted_ppr)
        # 三个独立的引入比例，sum 后作为总配额传给 core 模块
        total_extra = window_related_ratio + window_hub_ratio + window_random_ratio
        print(f"[分区] weightMetis_keepParent 构建分区 "
              f"(strategy={aug_strategy}, related={window_related_ratio:.0%}, "
              f"hub={window_hub_ratio:.0%}, random={window_random_ratio:.0%}, "
              f"total_extra={total_extra:.0%}) ...")
        wm = weightMetis_keepParent(
            csr_adjacency=csr_adj,
            eweights=eweights,
            edge_index=edge_index,
            edge_csr_data=None,
            n_parts=n_parts,
            attn_type="full",
            sorted_ppr_matrix=sorted_ppr,
            window_aug_strategy=aug_strategy,
            window_extra_node_ratio=total_extra,
            window_related_ratio=window_related_ratio,
            window_hub_ratio=window_hub_ratio,
            seed=seed,
        )
        partitions = wm.partitioned_results

    for i, p in enumerate(partitions):
        print(f"  分区 {i}: {len(p)} 个节点")
    return partitions


# ============================================================================
# 第 7 部分：捕获率计算（核心逻辑）
# ============================================================================

def compute_structural_capture_rate(pair_scores, partitions, topk_ratio):
    """
    【结构捕获率】不训练分区模型，仅根据分区结构判断：
    全注意力中 top-k% 的高分对，两端点是否落在同一个分区内。

    Args:
        pair_scores: 对称化后的注意力权重矩阵 [N, N]
        partitions:  list[Tensor]，每个 Tensor 是分区内全局节点索引
        topk_ratio:  float，如 0.05 表示取 top-5%

    Returns:
        capture_rate: float，被捕获的高分对比例
        num_total:    int  ，高分对总数
        num_captured: int  ，被捕获的高分对数量
    """
    N = pair_scores.shape[0]
    device = pair_scores.device

    # ---- 1. 取上三角（排除自环和重复） ----
    triu_mask = torch.triu(torch.ones(N, N, device=device), diagonal=1).bool()
    scores_upper = pair_scores[triu_mask]                    # [N*(N-1)/2]

    # ---- 2. 筛选 top-k% ----
    k = max(1, int(len(scores_upper) * topk_ratio))
    _, top_indices = torch.topk(scores_upper, k)

    # 扁平索引 → (row, col) 对
    rows, cols = torch.where(triu_mask)
    high_pairs = torch.stack([rows[top_indices], cols[top_indices]], dim=1)  # [k, 2]

    # ---- 3. 构建 节点 → 所在分区集合 的映射 ----
    node_to_parts = defaultdict(set)
    for part_id, nodes in enumerate(partitions):
        for n in nodes.tolist():
            node_to_parts[n].add(part_id)

    # ---- 4. 统计捕获数：两端点有共同分区即为捕获 ----
    captured = 0
    for i, j in high_pairs:
        i, j = i.item(), j.item()
        parts_i = node_to_parts.get(i, set())
        parts_j = node_to_parts.get(j, set())
        if parts_i & parts_j:  # 交集非空
            captured += 1

    return captured / k, k, captured


def compute_random_baseline(pair_scores, partitions, topk_ratio, n_trials=10):
    """
    【随机基线】将节点随机分配到与真实分区相同大小的桶中（不重叠）。
    重复 n_trials 次计算平均捕获率，作为基线对比。
    """
    N = pair_scores.shape[0]
    device = pair_scores.device
    all_nodes = torch.arange(N)
    partition_sizes = [len(p) for p in partitions]
    triu_mask = torch.triu(torch.ones(N, N, device=device), diagonal=1).bool()
    scores_upper = pair_scores[triu_mask]
    k = max(1, int(len(scores_upper) * topk_ratio))
    _, top_indices = torch.topk(scores_upper, k)
    rows, cols = torch.where(triu_mask)
    high_pairs_set = set()
    for idx in top_indices:
        high_pairs_set.add((rows[idx].item(), cols[idx].item()))

    rates = []
    for _ in range(n_trials):
        shuffled = all_nodes[torch.randperm(N)]
        random_parts = []
        start = 0
        for sz in partition_sizes:
            random_parts.append(set(shuffled[start:start + sz].tolist()))
            start += sz
        node_to_parts = defaultdict(set)
        for pid, nodes in enumerate(random_parts):
            for n in nodes:
                node_to_parts[n].add(pid)
        captured = 0
        for i, j in high_pairs_set:
            if node_to_parts.get(i, set()) & node_to_parts.get(j, set()):
                captured += 1
        rates.append(captured / len(high_pairs_set))
    return np.mean(rates), np.std(rates)


def compute_overlap_baseline(pair_scores, partitions, topk_ratio, n_trials=10):
    """
    【重叠随机基线】每个分区独立随机选等量节点（允许节点跨分区重复），
    保留与真实分区相同的重叠结构。用于对照扩展模式的分区重叠效应。
    """
    N = pair_scores.shape[0]
    device = pair_scores.device
    partition_sizes = [len(p) for p in partitions]
    triu_mask = torch.triu(torch.ones(N, N, device=device), diagonal=1).bool()
    scores_upper = pair_scores[triu_mask]
    k = max(1, int(len(scores_upper) * topk_ratio))
    _, top_indices = torch.topk(scores_upper, k)
    rows, cols = torch.where(triu_mask)
    high_pairs_set = set()
    for idx in top_indices:
        high_pairs_set.add((rows[idx].item(), cols[idx].item()))

    rates = []
    for _ in range(n_trials):
        node_to_parts = defaultdict(set)
        for pid, sz in enumerate(partition_sizes):
            selected = torch.randperm(N)[:sz].tolist()
            for n in selected:
                node_to_parts[n].add(pid)
        captured = 0
        for i, j in high_pairs_set:
            if node_to_parts.get(i, set()) & node_to_parts.get(j, set()):
                captured += 1
        rates.append(captured / len(high_pairs_set))
    return np.mean(rates), np.std(rates)


def compute_trained_capture_rate(model, feature, edge_index, partitions, pair_scores_full, topk_ratio, device):
    """
    【分区训练捕获率】在已训练收敛的分区模型上，检查全注意力高分对在分区
    注意力中是否也获得高分。

    流程：
      1. 从全注意力中取 top-k% 高分对，记入集合 S_full
      2. 对每个分区，取该分区内的注意力权重 [n_local, n_local]
      3. 对每个分区内，取 top-k% 局部分数 → 映射回全局 (i,j) → 记入 S_part
      4. 捕获率 = |S_full ∩ S_part| / |S_full|

    注意：分区模型的注意力仅在同窗口节点间计算，跨窗口对自动不被捕获。
    """
    model.eval()
    N = pair_scores_full.shape[0]

    # ---- 1. 全注意力高分对集合 ----
    triu_mask = torch.triu(torch.ones(N, N, device=pair_scores_full.device), diagonal=1).bool()
    scores_upper = pair_scores_full[triu_mask]
    k_full = max(1, int(len(scores_upper) * topk_ratio))
    _, top_indices = torch.topk(scores_upper, k_full)
    rows, cols = torch.where(triu_mask)
    full_high_set = set()
    for idx in top_indices:
        full_high_set.add((rows[idx].item(), cols[idx].item()))

    # ---- 2. 逐分区收集分区模型的高分对 ----
    part_high_set = set()

    for part_id, global_nodes in enumerate(partitions):
        if len(global_nodes) < 2:
            continue

        # 提取该分区的 pair_scores
        # 分区模型在 _pair_scores 中保存的是该窗口内的 [n_local, n_local]
        # 需要在每个分区独立前向
        x_part = feature[global_nodes].unsqueeze(0).to(device)
        model(x_part.squeeze(0), attn_bias=None, edge_index=None,
              in_degree=None, out_degree=None, spatial_pos=None, edge_input=None,
              attn_type="full")

        # 取该分区最后一层的 _pair_scores
        local_A = None
        for layer in model.layers:
            local_attn = layer.self_attention.local_attn
            if hasattr(local_attn, "_pair_scores") and local_attn._pair_scores is not None:
                local_A = local_attn._pair_scores.clone()
        if local_A is None:
            continue

        # 对称化
        local_A = (local_A + local_A.T) / 2.0
        n_local = local_A.shape[0]

        # 局部的 top-k% 对
        local_triu = torch.triu(torch.ones(n_local, n_local, device=local_A.device), diagonal=1).bool()
        local_scores = local_A[local_triu]
        k_local = max(1, int(len(local_scores) * topk_ratio))
        _, local_top_indices = torch.topk(local_scores, k_local)
        l_rows, l_cols = torch.where(local_triu)
        for idx in local_top_indices:
            li, lj = l_rows[idx].item(), l_cols[idx].item()
            gi = global_nodes[li].item()
            gj = global_nodes[lj].item()
            part_high_set.add((min(gi, gj), max(gi, gj)))

    # ---- 3. 交集 ----
    captured_set = full_high_set & part_high_set
    capture_rate = len(captured_set) / len(full_high_set) if full_high_set else 0.0
    return capture_rate, len(full_high_set), len(captured_set)


# ============================================================================
# 第 8 部分：分区模型训练
# ============================================================================

def train_partition_epochs(model, feature, y, edge_index, partitions,
                           split_idx, device, epochs, lr, weight_decay):
    """分区训练，跑满所有 epoch，不做早停。"""
    print(f"\n[分区训练] 在 {len(partitions)} 个窗口上训练...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_history = []
    best_val = -1.0

    for epoch in range(epochs):
        model.train()
        epoch_losses = []

        for part_id, global_nodes in enumerate(partitions):
            if len(global_nodes) < 2:
                continue

            x_part = feature[global_nodes].to(device)
            y_part = y[global_nodes].to(device)
            train_mask = torch.isin(global_nodes, split_idx["train"]).to(device)

            if not train_mask.any():
                continue

            optimizer.zero_grad()
            out, _, _, _ = model(
                x_part, attn_bias=None, edge_index=None,
                in_degree=None, out_degree=None, spatial_pos=None, edge_input=None,
                attn_type="full",
            )
            loss = F.nll_loss(out[train_mask], y_part[train_mask].long())
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
        loss_history.append(avg_loss)

        # 全图评估
        _, val_acc, test_acc = evaluate(model, feature, y, edge_index, split_idx, device, attn_type="full")
        if val_acc > best_val:
            best_val = val_acc

        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}: loss={avg_loss:.4f}, val_acc={val_acc:.4f}, test_acc={test_acc:.4f}")

    return model, loss_history, best_val


# ============================================================================
# 第 9 部分：绘图
# ============================================================================

def plot_results(results, args):
    """
    根据 results 中实际存在的数据动态绘图：
      - 有 structural_rates / trained_rates → 画捕获率柱状图
      - 有 full_loss / partition_loss → 画训练收敛曲线
    """
    os.makedirs(args.results_dir, exist_ok=True)
    ratios = [float(x) for x in args.topk_ratios.split(",")]
    x_labels = [f"{r*100:.0f}%" for r in ratios]
    x = np.arange(len(ratios))
    width = 0.35

    has_structural = bool(results.get("structural_rates"))
    has_trained = bool(results.get("trained_rates"))
    has_loss = "full_loss" in results or "partition_loss" in results

    # 决定子图数量：捕获率 + 损失曲线各占一列（如果都有的话）
    n_cols = (1 if (has_structural or has_trained) else 0) + (1 if has_loss else 0)
    if n_cols == 0:
        print("[图表] 无数据可供绘图，跳过。")
        return

    fig, axes = plt.subplots(1, n_cols, figsize=(7 * n_cols, 5))
    if n_cols == 1:
        axes = [axes]

    col_idx = 0

    # ---- 子图 A: 捕获率柱状图 ----
    if has_structural or has_trained:
        ax = axes[col_idx]
        col_idx += 1

        # 结构捕获率 vs 随机基线
        structural_rates = results.get("structural_rates", [])
        random_means = results.get("random_means", [])
        random_stds = results.get("random_stds", [])
        trained_rates = results.get("trained_rates", [])

        bar_positions = []
        bar_labels = []

        if structural_rates:
            bar_positions.append(x - width/2)
            bar_labels.append("结构捕获率 (Metis)")
            bars = ax.bar(x - width/2, structural_rates, width, label="结构捕获率 (Metis 分区)", color="#4C72B0")
            for bar, rate in zip(bars, structural_rates):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02,
                        f"{rate:.3f}", ha="center", va="bottom", fontsize=8)

        if random_means:
            bars = ax.bar(x + width/2, random_means, width, label="随机分区基线（不重叠）", color="#DD8452",
                          yerr=random_stds, capsize=4)
            for bar, rate in zip(bars, random_means):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02,
                        f"{rate:.3f}", ha="center", va="bottom", fontsize=8)

        # 重叠随机基线（允许节点跨分区重复）
        overlap_means = results.get("overlap_means", [])
        overlap_stds = results.get("overlap_stds", [])
        if overlap_means:
            bars = ax.bar(x + width/2, overlap_means, width / 2, label="重叠随机基线（跨分区重复）",
                          color="#E8B86D", yerr=overlap_stds, capsize=4)
            for bar, rate in zip(bars, overlap_means):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02,
                        f"{rate:.3f}", ha="center", va="bottom", fontsize=6)

        if trained_rates and not structural_rates:
            bars = ax.bar(x, trained_rates, width * 1.5, label="分区训练捕获率", color="#55A868")
            for bar, rate in zip(bars, trained_rates):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02,
                        f"{rate:.3f}", ha="center", va="bottom", fontsize=8)

        ax.set_xlabel("Top-k% 阈值")
        ax.set_ylabel("捕获率")
        ax.set_title(f"分区对全注意力高分对的捕获率 ({args.dataset})")
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    # ---- 子图 B: 训练收敛曲线 ----
    if has_loss:
        ax = axes[col_idx]
        if "full_loss" in results:
            ax.plot(results["full_loss"], label="全注意力训练", alpha=0.7, linewidth=1)
        if "partition_loss" in results:
            ax.plot(results["partition_loss"], label="分区训练", alpha=0.7, linewidth=1)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(f"训练损失曲线 ({args.dataset})")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(args.results_dir, f"capture_rate_{args.dataset}.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n[图表] 已保存到 {save_path}")
    plt.close(fig)


# ============================================================================
# 第 10 部分：主流程
# ============================================================================

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[设备] 使用 {device}")

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---- 加载数据 ----
    feature, y, edge_index, num_classes = load_dataset(args.dataset, args.dataset_dir)

    # ---- 大图子图采样 ----
    if args.dataset in _LARGE_DATASETS:
        feature, y, edge_index, num_classes = sample_subgraph(
            feature, y, edge_index,
            target_nodes=args.subgraph_nodes,
            n_hops=args.sample_hops,
            seed=args.seed,
        )

    N = feature.shape[0]
    split_idx = random_split_idx(y, seed=args.seed)

    # ---- 构建分区 ----
    partitions = build_partitions(edge_index, feature, args.n_parts, N, args.topk,
                                   expand=args.expand,
                                   aug_strategy=args.aug_strategy,
                                   window_related_ratio=args.window_related_ratio,
                                   window_hub_ratio=args.window_hub_ratio,
                                   window_random_ratio=args.window_random_ratio,
                                   seed=args.seed)

    results = {}

    # ========================================================================
    # 模式 A: 全注意力训练 + 结构捕获率
    # ========================================================================
    if args.mode in ("structural", "both"):
        print("\n" + "=" * 60)
        print("步骤 1/3: 训练全注意力模型")
        print("=" * 60)
        model_full = build_full_attention_model(feature.shape[1], num_classes, args).to(device)
        model_full, full_loss, full_val_acc = train_full_epochs(
            model_full, feature, y, edge_index, split_idx, device,
            attn_type="full", epochs=args.epochs_full, lr=args.lr,
            weight_decay=args.weight_decay,
        )
        results["full_loss"] = full_loss
        results["full_val_acc"] = full_val_acc

        print("\n" + "=" * 60)
        print("步骤 2/3: 提取对级注意力权重")
        print("=" * 60)
        pair_scores = extract_pair_scores(model_full, feature, edge_index, device)
        print(f"  注意力矩阵 shape: {pair_scores.shape}, "
              f"值域 [{pair_scores.min():.6f}, {pair_scores.max():.6f}]")

        print("\n" + "=" * 60)
        print("步骤 3/3: 计算结构捕获率")
        print("=" * 60)
        ratios = [float(x) for x in args.topk_ratios.split(",")]
        structural_rates = []
        random_means = []
        random_stds = []
        overlap_means = []
        overlap_stds = []
        for r in ratios:
            rate, total, captured = compute_structural_capture_rate(pair_scores, partitions, r)
            rand_mean, rand_std = compute_random_baseline(pair_scores, partitions, r)
            over_mean, over_std = compute_overlap_baseline(pair_scores, partitions, r)
            structural_rates.append(rate)
            random_means.append(rand_mean)
            random_stds.append(rand_std)
            overlap_means.append(over_mean)
            overlap_stds.append(over_std)
            print(f"  top-{r*100:5.1f}%: 结构捕获率={rate:.4f} ({captured}/{total}), "
                  f"随机基线(不重叠)={rand_mean:.4f}±{rand_std:.4f}, "
                  f"重叠基线={over_mean:.4f}±{over_std:.4f}")

        results["structural_rates"] = structural_rates
        results["random_means"] = random_means
        results["random_stds"] = random_stds
        results["overlap_means"] = overlap_means
        results["overlap_stds"] = overlap_stds

    # ========================================================================
    # 模式 B: 分区训练 + 捕获率
    # ========================================================================
    if args.mode in ("trained", "both"):
        print("\n" + "=" * 60)
        print("步骤: 训练分区模型")
        print("=" * 60)

        # 如果已从模式 A 中得到 pair_scores，复用之；否则先训全注意力模型
        if "structural_rates" not in results:
            print("  先训练全注意力模型以获取高分对...")
            model_full = build_full_attention_model(feature.shape[1], num_classes, args).to(device)
            model_full, full_loss, _ = train_full_epochs(
                model_full, feature, y, edge_index, split_idx, device,
                attn_type="full", epochs=args.epochs_full, lr=args.lr,
                weight_decay=args.weight_decay,
            )
            pair_scores = extract_pair_scores(model_full, feature, edge_index, device)
            results["full_loss"] = full_loss

        # 重新初始化一个模型用于分区训练
        model_part = build_full_attention_model(feature.shape[1], num_classes, args).to(device)
        model_part, part_loss, part_val_acc = train_partition_epochs(
            model_part, feature, y, edge_index, partitions, split_idx, device,
            epochs=args.epochs_partition, lr=args.lr, weight_decay=args.weight_decay,
        )
        results["partition_loss"] = part_loss
        results["partition_val_acc"] = part_val_acc

        print("\n" + "=" * 60)
        print("步骤: 计算分区训练捕获率")
        print("=" * 60)
        ratios = [float(x) for x in args.topk_ratios.split(",")]
        trained_rates = []
        for r in ratios:
            rate, total, captured = compute_trained_capture_rate(
                model_part, feature, edge_index, partitions, pair_scores, r, device,
            )
            trained_rates.append(rate)
            print(f"  top-{r*100:5.1f}%: 分区训练捕获率={rate:.4f} ({captured}/{total})")

        results["trained_rates"] = trained_rates

    # ---- 输出汇总 ----
    print("\n" + "=" * 60)
    print("实验汇总")
    print("=" * 60)
    if "full_val_acc" in results:
        print(f"  全注意力最佳验证准确率: {results['full_val_acc']:.4f}")
    if "partition_val_acc" in results:
        print(f"  分区训练最佳验证准确率: {results['partition_val_acc']:.4f}")

    # ---- 绘图 ----
    plot_results(results, args)


if __name__ == "__main__":
    main()
