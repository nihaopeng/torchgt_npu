# Baseline 多机每机一卡

所有命令都在控制节点的 `Baseline/scripts/cluster` 目录执行。每台机器使用
`CUDA_VISIBLE_DEVICES=0`，默认 Python 为 `/home/miniconda3/envs/gt/bin/python`，
数据目录为 `/home/dataset`。

## 最简流程

1. 编辑对应的 `hosts.N`，每行一个 IP，第一行是 master。
2. 同步整个仓库：

```bash
bash sync_code.sh 4
```

3. 后台启动：

```bash
bash launch.sh 4 arxiv
```

4. 等待结束并收集所有节点日志：

```bash
bash wait.sh 4 arxiv
```

5. 如需停止：

```bash
bash stop.sh 4
```

日志位于 `Baseline/TorchGT_logs/cluster/<数据集>/<N>card/`。也可在训练中手动执行
`bash collect_logs.sh 4 arxiv` 拉取当前日志。

## 示例

四卡 arxiv：

```bash
bash sync_code.sh 4
bash launch.sh 4 arxiv
bash wait.sh 4 arxiv
```

双卡 reddit：

```bash
bash sync_code.sh 2
bash launch.sh 2 reddit
bash wait.sh 2 reddit
```

支持 `arxiv`、`reddit`、`products`、`papers`，以及
`hosts.1`、`hosts.2`、`hosts.4`、`hosts.8`、`hosts.16`。
`papers` 还要求数据目录中存在 `split_idx.pt`。

默认 `num_heads=16`、`reorder=0`、`epochs=20`。环境变量可覆盖：

```bash
NUM_HEADS=8 REORDER=1 EPOCHS=30 SEQ_LEN=128000 bash launch.sh 4 arxiv
```

`REORDER` 仅接受 `0` 或 `1`；设为 `1` 时启动命令追加 `--reorder`。
默认 master 端口为 `29601`，可通过 `MASTER_PORT` 覆盖。

## SSH 指纹变化

机器重装或 IP 复用后，先确认新指纹可信，再清除控制节点上的旧记录：

```bash
NODE_IP=172.18.43.140
ssh-keygen -R "${NODE_IP}"
ssh "root@${NODE_IP}"
```

集群脚本不写入 `known_hosts`，但首次运行前仍建议人工核对新指纹。
