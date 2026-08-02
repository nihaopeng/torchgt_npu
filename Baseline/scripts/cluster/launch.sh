#!/usr/bin/env bash
# Launch Baseline with one GPU per node.
# Usage: bash launch.sh <1|2|4|8|16|hosts_file> <arxiv|reddit|products|papers>
set -euo pipefail

CLUSTER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINE_ROOT="$(cd "${CLUSTER_DIR}/../.." && pwd)"
REPO_ROOT="$(cd "${BASELINE_ROOT}/.." && pwd)"
# shellcheck source=lib.sh
source "${CLUSTER_DIR}/lib.sh"

if [[ $# -ne 2 ]]; then
  echo "Usage: bash $0 <1|2|4|8|16|hosts_file> <arxiv|reddit|products|papers>" >&2
  exit 1
fi

HOSTS_ARG="$1"
DATASET_ALIAS="$2"
EPOCHS="${EPOCHS:-20}"
NUM_HEADS="${NUM_HEADS:-16}"
REORDER="${REORDER:-0}"
SSH_KEY="${SSH_KEY:-/root/.ssh/id_ed25519}"
NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
MASTER_PORT="${MASTER_PORT:-29601}"
DATASET_DIR="${DATASET_DIR:-/home/dataset}"
PY="${PY:-/home/miniconda3/envs/gt/bin/python}"
LOG_ROOT="${LOG_ROOT:-${BASELINE_ROOT}/TorchGT_logs/cluster}"

if [[ ! "${NUM_HEADS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "NUM_HEADS must be a positive integer: ${NUM_HEADS}" >&2
  exit 1
fi
if [[ "${REORDER}" != "0" && "${REORDER}" != "1" ]]; then
  echo "REORDER must be 0 or 1: ${REORDER}" >&2
  exit 1
fi

resolve_hosts_file "${CLUSTER_DIR}" "${HOSTS_ARG}"
resolve_dataset "${DATASET_ALIAS}"
DATASET_ALIAS="$(normalize_dataset_alias "${DATASET_ALIAS}")"
load_hosts "${HOSTS_FILE}"
default_ssh_opts

if [[ "${HOSTS_ARG}" =~ ^(1|2|4|8|16)$ ]] && [[ "${NNODES}" -ne "${HOSTS_ARG}" ]]; then
  echo "ERROR: hosts.${HOSTS_ARG} has ${NNODES} IP(s), expected ${HOSTS_ARG}." >&2
  exit 1
fi

MASTER_ADDR="${HOSTS[0]}"
RUN_TAG="baseline_${DATASET_ALIAS}_${NNODES}card_$(date +%Y%m%d_%H%M%S)"

echo "============================================================"
echo "Baseline cluster dataset=${DATASET} model=graphormer attn_type=sparse"
echo "cards=${NNODES} hosts=${HOSTS_FILE} epochs=${EPOCHS}"
echo "master=${MASTER_ADDR}:${MASTER_PORT}"
echo "seq_len=${SEQ_LEN} num_heads=${NUM_HEADS} reorder=${REORDER}"
echo "baseline_root=${BASELINE_ROOT} repo_root=${REPO_ROOT}"
echo "run_tag=${RUN_TAG}"
echo "============================================================"

echo "[1/2] Preflight"
for RANK in "${!HOSTS[@]}"; do
  HOST="${HOSTS[$RANK]}"
  ssh "${SSH_OPTS[@]}" "root@${HOST}" \
    "test -x '${PY}' &&
     test -d '${REPO_ROOT}' &&
     test -f '${BASELINE_ROOT}/main_sp_node_level.py' &&
     test -d '${DATASET_DIR}/${DATASET}' &&
     test -e '${DATASET_DIR}/${DATASET}/x.pt' &&
     test -e '${DATASET_DIR}/${DATASET}/y.pt' &&
     test -e '${DATASET_DIR}/${DATASET}/edge_index.pt' &&
     nvidia-smi -L >/dev/null"
  if [[ "${DATASET}" == "ogbn-papers100M" ]]; then
    ssh "${SSH_OPTS[@]}" "root@${HOST}" \
      "test -e '${DATASET_DIR}/${DATASET}/split_idx.pt'"
  fi
  echo "  OK rank=${RANK} host=${HOST}"
done

echo "[2/2] Launch"
for RANK in "${!HOSTS[@]}"; do
  HOST="${HOSTS[$RANK]}"
  ssh "${SSH_OPTS[@]}" "root@${HOST}" bash -s -- \
    "${RANK}" "${NNODES}" "${MASTER_ADDR}" "${MASTER_PORT}" \
    "${DATASET}" "${DATASET_ALIAS}" "${SEQ_LEN}" "${EPOCHS}" "${RUN_TAG}" \
    "${NCCL_SOCKET_IFNAME}" "${DATASET_DIR}" "${BASELINE_ROOT}" "${PY}" \
    "${LOG_ROOT}" "${NUM_HEADS}" "${REORDER}" <<'REMOTE'
set -euo pipefail
RANK="$1"; NNODES="$2"; MASTER_ADDR="$3"; MASTER_PORT="$4"
DATASET="$5"; DATASET_ALIAS="$6"; SEQ_LEN="$7"; EPOCHS="$8"; RUN_TAG="$9"
NCCL_SOCKET_IFNAME="${10}"; DATASET_DIR="${11}"; BASELINE_ROOT="${12}"
PY="${13}"; LOG_ROOT="${14}"; NUM_HEADS="${15}"; REORDER="${16}"

LOG_DIR="${LOG_ROOT}/${DATASET_ALIAS}/${NNODES}card"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/node${RANK}_${RUN_TAG}.log"
PID_FILE="${LOG_DIR}/node${RANK}_${RUN_TAG}.pid"

export CUDA_VISIBLE_DEVICES=0
export PATH="$(dirname "${PY}"):/usr/local/cuda/bin:${PATH:-}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export NCCL_SOCKET_IFNAME
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

EXTRA_ARGS=()
if [[ "${REORDER}" == "1" ]]; then
  EXTRA_ARGS+=(--reorder)
fi

cd "${BASELINE_ROOT}"
printf 'Baseline config: dataset=%s seq_len=%s num_heads=%s reorder=%s epochs=%s\n' \
  "${DATASET}" "${SEQ_LEN}" "${NUM_HEADS}" "${REORDER}" "${EPOCHS}" > "${LOG_FILE}"
nohup "${PY}" -m torch.distributed.run \
  --nnodes="${NNODES}" --nproc_per_node=1 --node_rank="${RANK}" \
  --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" \
  main_sp_node_level.py \
  --dataset "${DATASET}" --dataset_dir "${DATASET_DIR}/" \
  --model graphormer --attn_type sparse \
  --seq_len "${SEQ_LEN}" --n_layers 4 --hidden_dim 64 --ffn_dim 64 \
  --num_heads "${NUM_HEADS}" --epochs "${EPOCHS}" \
  "${EXTRA_ARGS[@]}" \
  --distributed-backend nccl --distributed-timeout-minutes 120 \
  >>"${LOG_FILE}" 2>&1 &

echo "$!" > "${PID_FILE}"
echo "started rank=${RANK} pid=$! num_heads=${NUM_HEADS} reorder=${REORDER} log=${LOG_FILE}"
REMOTE
done

STATE_FILE="${LOG_ROOT}/state_baseline_${DATASET_ALIAS}_${NNODES}card.env"
mkdir -p "${LOG_ROOT}"
printf 'DATASET_ALIAS=%q\nNNODES=%q\nMASTER_ADDR=%q\nRUN_TAG=%q\nLOG_ROOT=%q\nBASELINE_ROOT=%q\n' \
  "${DATASET_ALIAS}" "${NNODES}" "${MASTER_ADDR}" "${RUN_TAG}" "${LOG_ROOT}" "${BASELINE_ROOT}" \
  > "${STATE_FILE}"

echo "Started: ${RUN_TAG}"
echo "Master log: ${LOG_ROOT}/${DATASET_ALIAS}/${NNODES}card/node0_${RUN_TAG}.log"
echo "Wait: bash ${CLUSTER_DIR}/wait.sh ${HOSTS_ARG} ${DATASET_ALIAS}"
