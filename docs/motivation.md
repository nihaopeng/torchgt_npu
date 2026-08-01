# 动机实验：注意力分数从 Q*K^T 改为 post-softmax 注意力权重

## 实验背景

NeutronGT 通过 Metis + PPR 将图划分为多个窗口（分区），在每个窗口内独立计算注意力，以 scale 到超大数据集。为了验证分区方案的可行性，需要证明：**全注意力（full attention）下的高分顶点对，在分区方案中也能被捕获**。

## 修改动机

原始代码中，`CoreAttention.full_attention()` 将注意力"分数"定义为 **softmax 之前的原始 Q*K^T 值**：

```python
score = torch.matmul(q, k)  # Q*K^T，值域 (-∞, +∞)，未归一化
```

这不够合理，因为：

1. Q*K^T 未归一化，不同节点/不同层的分数量纲不一致，难以比较
2. 未包含 attn_bias 和 mask 的影响
3. 最终决定注意力分配的实际上是 softmax 之后的权重

因此将 score 改为 **post-softmax 注意力权重**：

```python
x = torch.softmax(x, dim=3)  # 先归一化
score = x                     # 再捕获，值域 [0, 1]，行归一化
```

## 修改内容

### 文件 1：`NeutronGT/models/gt_dist_node_level_single_window.py`

`CoreAttention.full_attention()` 方法中：
- 删除第 78 行的 `score = x`（原 Q*K^T 捕获点）
- 在第 102 行 `x = torch.softmax(x, dim=3)` 之后添加 `score = x`

```diff
         q = q * self.scale
         x = torch.matmul(q, k)  # [b, num_head, seq_len, seq_len]
         log(f"x shape:{x.shape}")
-        score = x
         if attn_bias is not None:
             x = x + attn_bias
         ...
         x = torch.softmax(x, dim=3)
+        score = x
         x = self.att_dropout(x)
```

### 文件 2：`NeutronGT/models/graphormer_dist_node_level.py`

`CoreAttention.full_attention()` 方法中：
- 删除第 49 行的 `score = x`（原 Q*K^T 捕获点）
- 在第 60 行 `x = torch.softmax(x, dim=3)` 之后添加 `score = x`

```diff
         x = torch.matmul(q, k)
-        score = x

         if attn_bias is not None:
             ...
         x = torch.softmax(x, dim=3)
+        score = x
         x = self.att_dropout(x)
```

## 对下游的影响

| 组件 | 影响 |
|------|------|
| `node_out()` in `metisPartition.py` | 无。`node_scores` 的 shape `[N]` 不变，聚合逻辑不变 |
| `sparse_attention_bias()` | 无。稀疏注意力本身已用 exp 归一化 |
| `dist_attn` in `gt_layer.py` | 无。已支持 `context_layer, score_layer` 双返回值 |
| `vis.py` 可视化 | 无。仍以注意力矩阵为输入 |

## 对比

| 维度 | 修改前（Q*K^T） | 修改后（post-softmax） |
|------|----------------|----------------------|
| 值域 | (-∞, +∞) | [0, 1] |
| 行归一化 | 否 | 是 (每行 sum=1) |
| 包含 bias | 否 | 是 |
| 包含 mask | 否 | 是 |
| 语义 | 原始相关性 | 最终注意力分布 |

## 实验流程

1. 全注意力下训练模型至收敛（`attn_type="full"`）
2. 从最后一层（或所有层取平均）提取 post-softmax 注意力权重矩阵 `W ∈ [N, N]`
3. 聚合多头：对所有 head 取平均 → 得到 `[N, N]` 的对级分数
4. 设定阈值（如 top-5%）筛选高分顶点对
5. 对同一图运行分区方案，统计：
   - **捕获率** = 全注意力高分对中被分到同一窗口内的比例
   - **覆盖率** = 每个顶点的高分邻居被捕获的比例
6. 对比随机分区 baseline，验证分区方案的有效性

## 修改日期

2026-07-19
