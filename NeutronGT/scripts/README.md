# NeutronGT Scripts

本目录包含 NeutronGT 项目的训练启动脚本和实验工具。

## 目录

| 脚本 | 类型 | 说明 |
|------|------|------|
| [run_NeutronGT.sh](#run_neutrongtsh) | Shell | 主训练脚本（稀疏注意力 + KV Cache） |
| [run_ablation_2.sh](#run_ablation_2sh) | Shell | 消融实验：全注意力（无 Cache） |
| [run_ablation_3.sh](#run_ablation_3sh) | Shell | 消融实验：稀疏注意力（带 Cache） |
| [run_Runtimebreakdown.sh](#run_runtimebreakdownsh) | Shell | 运行时拆分实验（大模型） |
| [motivation_capture_rate.py](#motivation_capture_ratepy) | Python | 动机实验：分区方案对全注意力高分对的捕获率 |

---

## run_NeutronGT.sh

> **主训练入口**。使用 PPR + Metis 分区，在每个窗口内独立计算稀疏注意力，支持多 GPU 分布式训练（torchrun）。

**用法：**
```bash
bash scripts/run_NeutronGT.sh <CUDA_VISIBLE_DEVICES> --<dataset> --<model>
```

**参数：**
- `CUDA_VISIBLE_DEVICES`：必选，GPU 编号，如 `0,1,2,3`
- `--arxiv | --amazon | --reddit | --products`：数据集
- `--GT | --GPH_Slim | --GPH_Large`：模型

**示例：**
```bash
# 4 GPU 训练 GT 模型 on ogbn-arxiv
bash scripts/run_NeutronGT.sh 0,1,2,3 --arxiv --GT

# 8 GPU 训练 Graphormer Slim on AmazonProducts
bash scripts/run_NeutronGT.sh 0,1,2,3,4,5,6,7 --amazon --GPH_Slim

# 单 GPU 训练
bash scripts/run_NeutronGT.sh 0 --arxiv --GPH_Slim
```

**模型配置速查：**

| 别名 | model 参数 | layers | hidden | heads | attn | epochs |
|------|-----------|--------|--------|-------|------|--------|
| GT | `gt_sw` | 4 | 128 | 8 | sparse | 500 |
| GPH_Slim | `graphormer` | 4 | 64 | 8 | sparse | 500 |
| GPH_Large | `graphormer` | 12 | 768 | 32 | sparse | 200 |

**输出：** 日志保存在 `NeutronGT_logs/` 目录下，文件名包含数据集、模型、时间戳。

---

## run_ablation_2.sh

> **消融实验 #2**：验证全注意力（full attention）的效果。使用 Graphormer Slim + **full attn** + **无 KV Cache**。与 run_ablation_3.sh 对照，对比 full vs sparse attention。

**用法：**
```bash
bash scripts/run_ablation_2.sh <CUDA_VISIBLE_DEVICES> --<dataset> --GPH_Slim
```

**关键差异：**
- `attn_type="full"`（全注意力，计算 O(N²) 的注意力矩阵）
- `use_cache=0`（不使用重复节点 KV 缓存）
- 25 epochs（仅验证趋势，不追求收敛）

**示例：**
```bash
bash scripts/run_ablation_2.sh 0,1,2,3 --arxiv --GPH_Slim
```

---

## run_ablation_3.sh

> **消融实验 #3**：验证稀疏注意力 + KV Cache 的效果。使用 Graphormer Slim + **sparse attn** + **带 KV Cache**。与 run_ablation_2.sh 对照。

**用法：**
```bash
bash scripts/run_ablation_3.sh <CUDA_VISIBLE_DEVICES> --<dataset> --GPH_Slim
```

**关键差异：**
- `attn_type="sparse"`（仅沿图边计算注意力）
- `use_cache=1`（启用重复节点 KV 缓存，避免重复计算）

**示例：**
```bash
bash scripts/run_ablation_3.sh 0,1,2,3 --arxiv --GPH_Slim
```

**对照表（ablation_2 vs ablation_3）：**

| 维度 | run_ablation_2 | run_ablation_3 |
|------|---------------|---------------|
| 注意力类型 | full (O(N²)) | sparse (O(E)) |
| KV Cache | 关闭 | 开启 |
| 内存 | 高 | 低 |
| 目的 | 验证全注意力基线 | 验证分区+稀疏方案 |

---

## run_Runtimebreakdown.sh

> **运行时拆分实验**：使用大模型（GPH_Large, 12层 768维 32头）训练 500 epochs，测量各阶段耗时（CPU→GPU、前向/反向、通信等），用于分析训练瓶颈。

**用法：**
```bash
bash scripts/run_Runtimebreakdown.sh <CUDA_VISIBLE_DEVICES> --<dataset> --GPH_Large
```

**示例：**
```bash
bash scripts/run_Runtimebreakdown.sh 0,1,2,3,4,5,6,7 --arxiv --GPH_Large
```

---

## motivation_capture_rate.py

> **动机实验**：验证分区方案对全注意力高分顶点对的捕获能力。核心思路：先在完整图上跑全注意力训练至收敛，提取高注意力顶点对，然后检查这些高分对是否落在同一分区窗口中。

**用法：**
```bash
python scripts/motivation_capture_rate.py [OPTIONS]
```

**核心参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset` | `cora` | 数据集：cora / citeseer / pubmed / ogbn-arxiv / reddit / ogbn-products / amazon / papers100 |
| `--mode` | `both` | 实验模式：`structural` / `trained` / `both` |
| `--n_parts` | `4` | 分区数量（需为偶数，扩展模式父子分区各一半） |
| `--expand` | `False` | 启用分区增广（与训练流程一致：边界邻居 + hub 高度节点 + 随机填充） |
| `--aug_strategy` | `ours` | 增广策略：`ours`=related+hub+随机 / `hub`=仅高度节点 / `related`=仅边界邻居 / `random`=仅随机 |
| `--window_related_ratio` | `0.15` | 边界邻居引入比例（相对核心分区大小，独立于其他来源） |
| `--window_hub_ratio` | `0.15` | hub 高度节点引入比例（相对核心分区大小，独立于其他来源） |
| `--window_random_ratio` | `0.0` | 随机填充节点比例（相对核心分区大小，默认 0 避免抬高随机基线） |
| `--subgraph_nodes` | `3000` | 大图采样子图目标节点数（仅 ogbn-arxiv / reddit / ogbn-products / amazon / papers100） |
| `--sample_hops` | `2` | 子图采样 BFS 邻居扩展跳数 |
| `--n_layers` | `3` | Transformer 层数 |
| `--hidden_dim` | `128` | 隐藏层维度 |
| `--num_heads` | `4` | 注意力头数 |
| `--epochs_full` | `300` | 全注意力训练轮数 |
| `--epochs_partition` | `300` | 分区模型训练轮数 |
| `--lr` | `0.001` | 学习率 |
| `--topk_ratios` | `0.01,0.02,0.05,0.10,0.20` | 捕获率计算时的高分对阈值列表 |
| `--topk` | `64` | PPR top-k 邻居数 |
| `--dataset_dir` | 项目 `dataset/` | 数据集根目录（默认绝对路径，不受运行目录影响） |
| `--device` | `cuda` | 计算设备 |
| `--results_dir` | `./results/motivation` | 图表输出目录 |
| `--seed` | `42` | 随机种子 |

**数据集说明：**

| 数据集 | 节点数 | 来源 | 需要子图采样 |
|--------|--------|------|-------------|
| cora | 2,708 | 本地 .pt | 否 |
| citeseer | 3,327 | 本地 .pt | 否 |
| pubmed | 19,717 | 本地 .pt | 否 |
| amazon | 13,752 | PyG 自动下载 | 是 |
| ogbn-arxiv | 169,343 | OGB 自动下载 | 是 |
| reddit | 232,965 | PyG 自动下载 | 是 |
| ogbn-products | 2,449,029 | OGB 自动下载 | 是 |
| papers100 | 111,059,956 | OGB 自动下载 | 是 |

**三种实验模式：**

| 模式 | 含义 |
|------|------|
| `structural` | 全注意力训练 → 提取高分对 → 直接用分区结构算捕获率（不训分区模型） |
| `trained` | 全注意力训练 → 分区模型也训练 → 检查分区模型同样给高分对打高分 |
| `both` | 两者都跑，绘制对比图 |

**示例：**
```bash
# 小图快速验证
python scripts/motivation_capture_rate.py --dataset cora --mode structural --epochs_full 100

# 大图 + 子图采样 + 分区扩展（仅边界邻居，不引入随机填充）
python scripts/motivation_capture_rate.py --dataset amazon --mode structural \
    --subgraph_nodes 3000 --expand --aug_strategy related --window_related_ratio 0.6

# ogbn-arxiv：ours 策略，边界邻居 60% + hub 30%，无随机
python scripts/motivation_capture_rate.py --dataset ogbn-arxiv --mode both \
    --subgraph_nodes 3000 --topk 32 --expand \
    --window_related_ratio 0.6 --window_hub_ratio 0.3 --window_random_ratio 0.0 \
    --epochs_full 1000


# 仅 hub 高度节点引入
python scripts/motivation_capture_rate.py --dataset reddit --mode structural \
    --subgraph_nodes 3000 --expand --aug_strategy hub --window_hub_ratio 0.6

# 自定义阈值扫描
python scripts/motivation_capture_rate.py --dataset pubmed --mode structural \
    --topk_ratios "0.01,0.03,0.05,0.10,0.15,0.20" --n_parts 6
```

**输出：**
- `results/motivation/capture_rate_<dataset>.png` — 捕获率柱状图（分区方案 vs 随机基线）和训练损失收敛曲线

**输出指标解读：**
- **结构捕获率**：全注意力 top-k% 高分对中，两端点落在同一分区的比例
- **随机基线**：相同分区大小下随机分配节点的捕获率（≈ 1/n_parts for 等大分区）
- 若捕获率 > 随机基线 → 分区策略有效捕获了注意力高分对
- 若捕获率 < 随机基线 → 全注意力看重的关系超越了图拓扑所能表达的

**依赖：**
- PyTorch, pymetis, matplotlib
- PyG / OGB（大图自动下载）
- 项目内模块：`models/graphormer_dist_node_level`、`core/ppr_preprocess`、`core/metisPartition`（扩展模式）
