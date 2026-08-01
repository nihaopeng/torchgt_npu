#!/usr/bin/env python3
"""捕获率柱状图：top-1% 和 top-5% 并排，NeutronGT vs 随机基线。"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ============================================================
# 数据
# ============================================================
datasets = ["ogbn-arxiv", "papers100", "amazon", "ogbn-products", "reddit"]
x_labels = ["OAV", "OPR", "AZ", "OPT", "RDT"]

top1 = {
    "Minibatch":    [0.3424, 0.3420, 0.3429, 0.3449, 0.3442],
    "Window":       [0.5593, 0.3787, 0.4936, 0.4365, 0.4459],
}

top5 = {
    "Minibatch":    [0.3432, 0.3423, 0.3431, 0.3438, 0.3437],
    "Window":       [0.5316, 0.3934, 0.4945, 0.4284, 0.4682],
}

# ============================================================
# 风格参数
# ============================================================
plot_params = {
    "font.family": "Arial",
    "font.weight": "bold",
    "axes.labelsize": 24,
    "axes.titlesize": 27,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 20,
    "lines.linewidth": 1.2,
}

colors = ["#4C72B0", "#DD8452"]   # 蓝 / 橙
hatchs = ["", "//"]
bar_width = 0.30

# ============================================================
# 绘制
# ============================================================
plt.rcParams.update(plot_params)
plt.rcParams["pdf.fonttype"] = 42

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

handles, labels = [], []
for ax, (data_dict, sub_label) in zip(axes, [(top1, "(a) Top-1%"), (top5, "(b) Top-5%")]):
    n_groups = len(datasets)
    n_bars = len(data_dict)
    x = np.arange(n_groups)
    offset = (n_bars - 1) / 2

    for i, (label, values) in enumerate(data_dict.items()):
        pos = x + (i - offset) * bar_width
        bars = ax.bar(
            pos, values, bar_width,
            color=colors[i], hatch=hatchs[i],
            edgecolor="black", linewidth=0.8,
            zorder=3,
        )
        if label not in labels:
            handles.append(bars)
            labels.append(label)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("Capture Rate")
    ax.set_xlabel(sub_label, fontsize=24, fontweight="bold")
    ax.set_ylim(0, 0.60)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.grid(axis="y", linestyle="--", alpha=0.3, zorder=0)

# 共享图例，放在图顶部外侧
fig.legend(
    handles, labels,
    loc="upper center", ncol=len(labels),
    frameon=False,
    bbox_to_anchor=(0.5, 1.02),
    fontsize=20, columnspacing=1.5, handletextpad=0.6,
)

fig.tight_layout(pad=1.0, rect=(0, 0, 1, 0.93))
fig.savefig("capture_rate.pdf", dpi=300, bbox_inches="tight")
plt.close(fig)
print("[图表] 已保存 capture_rate.pdf")
