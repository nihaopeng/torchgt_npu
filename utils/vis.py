from collections import Counter
import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import networkx as nx

plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']

score_hist_flist = []
score_neighbor_ratio_list = []
score_relativity_ratio_list = []
node_attned_list = []
pers = [95,90,80,70,60,40,20]
epochs = []
train_acc = []
val_acc = []
test_acc = []

def pics_to_gif(flist,output_gif, frame_duration=100, loop=0):
    """
    读取文件夹中的图片生成GIF
    :param folder_path: 图片文件夹路径
    :param output_gif: 输出GIF的路径及文件名
    :param frame_duration: 每帧持续时间（毫秒），数值越小播放越快
    :param loop: 循环次数，0表示无限循环
    """
    # 获取文件夹中所有png/jpg格式的图片，并按文件名排序
    frames = []
    for img_path in flist:
        with Image.open(img_path) as img:
            # 转换为RGB模式，避免GIF不支持透明度导致的异常
            frames.append(img.convert('RGB'))
    # 保存为GIF
    frames[0].save(
        output_gif,
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration,
        loop=loop
    )
    print(f"GIF已保存至：{output_gif}")

def high_attn(score_agg:np.ndarray,score_spe:np.ndarray,epoch:int):
    plt.figure(figsize=(10, 6))
    n_bins = 50
    attention_flat = score_agg.flatten()
    counts, bins, patches = plt.hist(
        attention_flat,
        bins=n_bins,
        alpha=0.8,
        edgecolor="black",
        color="#1f77b4"
    )
    plt.xlim(0, 200)
    plt.title(f"注意力分数分布直方图 (n={score_agg.size})", fontsize=14, fontweight="bold")
    plt.xlabel("注意力分数值", fontsize=12)
    plt.ylabel("频数（像素/元素个数）", fontsize=12)
    plt.tight_layout()
    fig_full_path = os.path.join("vis","score_hist",f"attention_histogram_{epoch}.png")
    if not os.path.exists("./vis/score_hist"):
        os.mkdir("./vis/score_hist")
    plt.savefig(fig_full_path, dpi=300, bbox_inches="tight")
    plt.close()
    score_hist_flist.append(fig_full_path)

def plot(x,y,fig_name):
    fig, ax = plt.subplots(figsize=(12, 7))

    for p_idx, p in enumerate(pers):
        ratios = y[p_idx]
        ax.plot(epochs, ratios,
                label=f'{p}% 百分位数')
    ax.legend(
        loc='upper right',
        fontsize=10,
        ncol=2,
        frameon=True,
        shadow=True
    )
    ax.tick_params(axis='x', rotation=45)
    ax.set_xticks(epochs)
    plt.title(f"{fig_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f'{fig_name}_ratio_by_epoch.png', dpi=300, bbox_inches='tight')
    plt.close()

def neighbor(score_agg:np.ndarray,idx:np.ndarray,edge_index:np.ndarray,epoch:int):
    score_flat=score_agg.flatten()
    thresh_holds=np.percentile(score_flat,pers)
    
    edge_set = set()
    # print(f"edge_index.shape:{edge_index.shape}")
    for u, v in edge_index.T:
        edge_set.add((min(u, v), max(u, v)))
        
    score_neighbor_ratio_epoch = []
    for k,th in enumerate(thresh_holds):
        score_per_n = score_agg>th
        score_cnt = np.sum(score_per_n)
        score_neighbor_cnt = 0
        for i in range(score_agg.shape[0]):
            for j in range(score_agg.shape[1]):
                if i != j and score_per_n[i][j]:
                    # 获取对应的节点索引
                    u = idx[i]
                    v = idx[j]
                    # 检查是否在边集合中（O(1)查询）
                    edge_tuple = (min(u, v), max(u, v))
                    # print(f"edge_tuple:{edge_tuple}")
                    if edge_tuple in edge_set:
                        score_neighbor_cnt += 1
        score_neighbor_ratio_epoch.append(score_neighbor_cnt/score_cnt)
    score_neighbor_ratio_list.append(score_neighbor_ratio_epoch)

def relativity(score_agg:np.ndarray,idx:np.ndarray,edge_index:np.ndarray,epoch:int):
    score_flat=score_agg.flatten()
    thresh_holds=np.percentile(score_flat,pers)
    score_relativity_ratio_epoch = []
    for k,th in enumerate(thresh_holds):
        score_per_n = score_agg>th
        score_cnt = np.sum(score_per_n)
        score_relativity_cnt = 0
        for i in range(score_agg.shape[0]):
            for j in range(i+1,score_agg.shape[1]):
                if score_per_n[i][j] and score_per_n[j][i]:
                    if score_agg[i][j]//20==score_agg[j][i]//20:
                        score_relativity_cnt+=1
        score_relativity_ratio_epoch.append(score_relativity_cnt/score_cnt)
    score_relativity_ratio_list.append(score_relativity_ratio_epoch)

def high_attn_node_plot():
    if not os.path.exists("./vis/node_attned"):
        os.mkdir("./vis/node_attned")
    plt.figure(figsize=(10, 6))
    for peridx,pern in enumerate(pers):
        high_attn_flist = []
        for eidx,e in enumerate(epochs):
            plt.hist(
                node_attned_list[eidx][peridx],
                bins=50,
                alpha=0.8,
                edgecolor="black",
                color="#1f77b4"
            )
            plt.xlim(0, 2.5)
            plt.title(f"被高注意节点的高分占比比例", fontsize=14, fontweight="bold")
            plt.xlabel("比例值", fontsize=12)
            plt.ylabel("频数", fontsize=12)
            plt.tight_layout()
            fig_full_path = os.path.join("vis","node_attned",f"per_{pern}",f"attned_per{pern}_{e}.png")
            if not os.path.exists(f"./vis/node_attned/per_{pern}"):
                os.mkdir(f"./vis/node_attned/per_{pern}")
            plt.savefig(fig_full_path, dpi=300, bbox_inches="tight")
            plt.close()
            high_attn_flist.append(fig_full_path)
        pics_to_gif(high_attn_flist,f"./vis/high_attn_pern{pern}.gif")

def high_attn_node(score_agg:np.ndarray,idx:np.ndarray,edge_index:np.ndarray,epoch:int):
    score_flat=score_agg.flatten()
    thresh_holds=np.percentile(score_flat,pers)
    node_attned = []
    for k,th in enumerate(thresh_holds):
        node_attned_pern = []
        for i in range(score_agg.shape[0]):
            high_score_cnt = np.sum(score_agg[:, i] > th)
            node_attned_pern.append(high_score_cnt/score_agg.shape[1])
        node_attned.append(node_attned_pern)
    node_attned_list.append(node_attned)

def distance(score_agg:np.ndarray,idx:np.ndarray,edge_index:np.ndarray,epoch:int):
    import networkx as nx
    # 从edge_index构建图
    G = nx.Graph()
    edges = list(zip(edge_index[0], edge_index[1]))
    G.add_edges_from(edges)

    score_flat=score_agg.flatten()
    thresh_holds=np.percentile(score_flat,pers)
    for k,th in enumerate(thresh_holds):
        score_per_n = score_agg > th
        high_attention_indices = np.argwhere(score_per_n)
        distances = []
        for i, j in high_attention_indices:
            u = idx[i]
            v = idx[j]
            try:
                # 计算节点对的最短路径长度
                dis = nx.shortest_path_length(G, source=u, target=v)
                distances.append(dis)
            except nx.NetworkXNoPath:
                distances.append(0)
            except Exception as e:
                print(f"计算节点对 ({u}, {v}) 距离时出错: {e}")
                continue
        distance_counts = Counter(distances)
        sorted_distances = sorted(distance_counts.keys())
        counts = [distance_counts[d] for d in sorted_distances]
        plt.bar(
            sorted_distances,
            counts,
            alpha=0.8,
            edgecolor="black",
            color="#1f77b4",
            width=0.8
        )
        plt.xlim(0, 25)
        plt.title(f"高注意节点的距离统计", fontsize=14, fontweight="bold")
        plt.xlabel("距离", fontsize=12)
        plt.ylabel("频数", fontsize=12)
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        if not os.path.exists(f"./vis/dis/per{pers[k]}/"):
            os.makedirs(f"./vis/dis/per{pers[k]}/")
        fig_full_path = os.path.join("vis","dis",f"per{pers[k]}",f"distance_epoch{epoch}.jpg")
        plt.savefig(fig_full_path, dpi=300, bbox_inches="tight")
        plt.close()

def mask_high_attn(score_agg:np.ndarray,idx:np.ndarray,edge_index:np.ndarray,epoch:int) -> torch.Tensor:
    score_agg=score_agg.cpu().detach().numpy() if isinstance(score_agg,torch.Tensor) else score_agg
    idx = idx.cpu().detach().numpy() if isinstance(idx,torch.Tensor) else idx
    edge_index=edge_index.cpu().detach().numpy() if isinstance(edge_index,torch.Tensor) else edge_index
    if score_agg is not None:
        score_flat=score_agg.flatten()
        thresh_hold=np.percentile(score_flat,20) # 一次跑一个数据
        mask = score_agg<thresh_hold
        return torch.tensor(mask, dtype=torch.bool)
    else:
        return None
    
def acc_plot(x,y,fig_name):
    fig, ax = plt.subplots(figsize=(12, 7))
    for i,acc in enumerate(y):
        ax.plot(x, acc, label=f"{fig_name[i]}_acc")
    ax.legend(
        loc='upper left',
        fontsize=10,
        ncol=2,
        frameon=True,
        shadow=True
    )
    ax.set_ylim(0,1)
    ax.tick_params(axis='x', rotation=45)
    ax.set_xticks(epochs)
    plt.tight_layout()
    plt.savefig(f'acc_by_epoch.png', dpi=300, bbox_inches='tight')
    plt.close()

def homo_node_mask(edge_index, idx_i, mask_ratio=0.5):
    """
    极简版：用NetworkX筛选邻居，生成屏蔽掩码（True=屏蔽该边）
    """
    # 1. 转NetworkX图（无向）
    G = nx.Graph()
    G.add_edges_from(edge_index.T.cpu().numpy().tolist())
    
    mask = torch.zeros((len(idx_i), len(idx_i)), dtype=torch.bool)
    
    node_to_local_idx = {node: idx for idx, node in enumerate(idx_i.cpu().numpy())}
    for local_i,node in enumerate(idx_i.cpu().numpy()):
        # 找同质点邻居
        neighbors = [n for n in G.neighbors(node) if n in node_to_local_idx]
        # if len(neighbors) <= 1: continue

        np.random.shuffle(neighbors)
        mask_num = int(len(neighbors) * mask_ratio)
        masked_neis = neighbors[-mask_num:]

        # 标记掩码（无向图检查双向边）
        for nei in masked_neis:
            # 找到边在全局edge_index中的位置并设为True
            local_j = node_to_local_idx[nei]  # 邻居在idx_i中的局部位置
            mask[local_i, local_j] = True     # 标记屏蔽
    return mask

def vis_interface(score_agg,score_spe,idx,edge_index,epoch):
    score_agg=score_agg.cpu().detach().numpy() if isinstance(score_agg,torch.Tensor) else score_agg
    score_spe=score_spe.cpu().detach().numpy() if isinstance(score_spe,torch.Tensor) else score_spe
    idx = idx.cpu().detach().numpy() if isinstance(idx,torch.Tensor) else idx
    edge_index=edge_index.cpu().detach().numpy() if isinstance(edge_index,torch.Tensor) else edge_index
    epochs.append(epoch)
    if not os.path.exists("vis"):
        os.mkdir("vis")
    # high_attn(score_agg,score_spe,epoch)
    # neighbor(score_agg,idx,edge_index,epoch)
    # relativity(score_agg,idx,edge_index,epoch)
    # high_attn_node(score_agg,idx,edge_index,epoch)
    # distance(score_agg,idx,edge_index,epoch)
