# NeutronGT 多机多卡运行

当前方式是每台机器使用一张 GPU。所有命令都在控制节点执行。

## 运行前检查

- 控制节点可以通过 SSH 登录所有节点。
- 每台机器上的仓库路径相同。
- 每台机器都有相同的数据和 Python 环境。
- 默认 Python：`/home/miniconda3/envs/gt/bin/python`
- 默认数据目录：`/home/dataset`

## SSH 指纹不匹配

机器重装或 IP 被重新分配后，可能出现 IP 相同但 SSH 指纹变化：

```text
WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED!
```

先向机器管理员确认新指纹可信，再在控制节点删除该 IP 的旧记录：

```bash
NODE_IP=172.18.43.140
ssh-keygen -R "${NODE_IP}"
ssh "root@${NODE_IP}"
```

首次重新连接时核对并接受新指纹。集群脚本本身不会写入
`known_hosts`，但建议先完成上述人工验证，避免连接到错误的机器。

## 四卡 arxiv 示例

### 1. 配置节点

编辑 `hosts.4`，每行填写一台机器的 IP，第一行是主节点：

```text
主节点IP
节点1_IP
节点2_IP
节点3_IP
```

每台机器都需要以下文件：

```text
/home/dataset/ogbn-arxiv/x.pt
/home/dataset/ogbn-arxiv/y.pt
/home/dataset/ogbn-arxiv/edge_index.pt
/home/dataset/ogbn-arxiv/edge_index_csr.pt
```

### 2. 同步代码

首次运行或修改代码后执行：

```bash
cd /home/22222222/NeutronGT/NeutronGT/scripts/cluster
bash sync_code.sh 4
```

### 3. 启动训练

```bash
EPOCHS=20 bash launch.sh 4 arxiv
```

### 4. 等待结束并收集日志

```bash
bash wait.sh 4 arxiv
```

日志保存在：

```text
NeutronGT/NeutronGT_logs/cluster/arxiv/4card/
```

### 5. 停止任务

```bash
bash stop.sh 4
```

## 其他配置

双卡 Reddit 示例：

```bash
bash sync_code.sh 2
EPOCHS=20 bash launch.sh 2 reddit
bash wait.sh 2 reddit
```

数据集参数支持：`arxiv`、`reddit`、`products`、`papers`。卡数对应使用
`hosts.1`、`hosts.2`、`hosts.4`、`hosts.8` 或 `hosts.16`。

