#!/usr/bin/env bash
# Multi-node NeutronGT launcher (self-contained; no external scripts_dist).
#
# Usage:
#   bash launch.sh <1|2|4|8|16|hosts_file> <arxiv|reddit|products|papers>
#
# Env overrides:
#   EPOCHS=20 DATASET_DIR=/home/dataset LOG_ROOT=... SSH_KEY=... MASTER_PORT=29600
#   NCCL_SOCKET_IFNAME=eth0 USE_PREPROCESS_CACHE=1 REFRESH_PREPROCESS_CACHE=0
#   PY=/home/miniconda3/envs/gt/bin/python
set -euo pipefail

CLUSTER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEUTRON_ROOT="$(cd "${CLUSTER_DIR}/../.." && pwd)"
# shellcheck source=lib.sh
source "${CLUSTER_DIR}/lib.sh"

if [[ $# -ne 2 ]]; then
  echo "Usage: bash $0 <1|2|4|8|16|hosts_file> <arxiv|reddit|products|papers>" >&2
  exit 1
fi

HOSTS_ARG="$1"
DATASET_ALIAS="$2"
EPOCHS="${EPOCHS:-20}"
SSH_KEY="${SSH_KEY:-/root/.ssh/id_ed25519}"
NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
MASTER_PORT="${MASTER_PORT:-29600}"
USE_PREPROCESS_CACHE="${USE_PREPROCESS_CACHE:-1}"
REFRESH_PREPROCESS_CACHE="${REFRESH_PREPROCESS_CACHE:-0}"
WINDOW_ASSIGNMENT_STRATEGY="${WINDOW_ASSIGNMENT_STRATEGY:-edge_balanced_step}"
DATASET_DIR="${DATASET_DIR:-/home/dataset}"
PY="${PY:-/home/miniconda3/envs/gt/bin/python}"
LOG_ROOT="${LOG_ROOT:-${NEUTRON_ROOT}/NeutronGT_logs/cluster}"

resolve_hosts_file "${CLUSTER_DIR}" "${HOSTS_ARG}"
resolve_dataset "${DATASET_ALIAS}"
DATASET_ALIAS="$(normalize_dataset_alias "${DATASET_ALIAS}")"
load_hosts "${HOSTS_FILE}"
default_ssh_opts

if [[ "${HOSTS_ARG}" =~ ^(1|2|4|8|16)$ ]] && [[ "${NNODES}" -ne "${HOSTS_ARG}" ]]; then
  echo "ERROR: hosts.${HOSTS_ARG} has ${NNODES} IP(s), expected exactly ${HOSTS_ARG}." >&2
  echo "Edit ${HOSTS_FILE} (one IP per line, first = master)." >&2
  exit 1
fi

MASTER_ADDR="${HOSTS[0]}"
RUN_TAG="neutron_${DATASET_ALIAS}_${NNODES}card_$(date +%Y%m%d_%H%M%S)"
WRAP_SRC="${CLUSTER_DIR}/wrap_neutron_main.py"

echo "============================================================"
echo "NeutronGT cluster  dataset=${DATASET} model=GPH_Slim epochs=${EPOCHS}"
echo "cards=${NNODES} hosts=${HOSTS_FILE}"
echo "master=${MASTER_ADDR}:${MASTER_PORT}"
echo "n_parts=${NPARTS} window=${WINDOW_EXTRA_RATIO}/${WINDOW_RELATED_RATIO}/${WINDOW_HUB_RATIO}"
echo "window_assignment=${WINDOW_ASSIGNMENT_STRATEGY}"
echo "dataset_dir=${DATASET_DIR} neutron_root=${NEUTRON_ROOT}"
echo "run_tag=${RUN_TAG}"
echo "============================================================"

echo "[1/2] Preflight"
for RANK in "${!HOSTS[@]}"; do
  HOST="${HOSTS[$RANK]}"
  ssh "${SSH_OPTS[@]}" "root@${HOST}" \
    "test -x '${PY}' &&
     test -d '${NEUTRON_ROOT}' &&
     test -f '${NEUTRON_ROOT}/main_sp_node_level_ppr.py' &&
     test -d '${DATASET_DIR}/${DATASET}' &&
     test -e '${DATASET_DIR}/${DATASET}/x.pt' &&
     test -e '${DATASET_DIR}/${DATASET}/y.pt' &&
     test -e '${DATASET_DIR}/${DATASET}/edge_index.pt' &&
     test -e '${DATASET_DIR}/${DATASET}/edge_index_csr.pt' &&
     nvidia-smi -L >/dev/null"
  if [[ "${NNODES}" -gt 1 ]]; then
    scp "${SSH_OPTS[@]}" "${WRAP_SRC}" \
      "root@${HOST}:${NEUTRON_ROOT}/wrap_neutron_main.py" >/dev/null
  fi
  echo "  OK rank=${RANK} host=${HOST}"
done

echo "[2/2] Launch"
for RANK in "${!HOSTS[@]}"; do
  HOST="${HOSTS[$RANK]}"
  PEER_HOSTS=""
  if [[ "${NNODES}" -gt 1 ]]; then
    for P in "${!HOSTS[@]}"; do
      [[ "${P}" -eq "${RANK}" ]] && continue
      [[ -n "${PEER_HOSTS}" ]] && PEER_HOSTS+=","
      PEER_HOSTS+="${HOSTS[$P]}"
    done
  fi

  ssh "${SSH_OPTS[@]}" "root@${HOST}" bash -s -- \
    "${RANK}" "${NNODES}" "${MASTER_ADDR}" "${MASTER_PORT}" \
    "${DATASET}" "${DATASET_ALIAS}" "${NPARTS}" "${EPOCHS}" "${RUN_TAG}" \
    "${NCCL_SOCKET_IFNAME}" "${PEER_HOSTS:--}" "${SSH_KEY}" \
    "${DATASET_DIR}" "${NEUTRON_ROOT}" "${PY}" "${LOG_ROOT}" \
    "${WINDOW_EXTRA_RATIO}" "${WINDOW_RELATED_RATIO}" "${WINDOW_HUB_RATIO}" \
    "${USE_PREPROCESS_CACHE}" "${REFRESH_PREPROCESS_CACHE}" "${SEQ_LEN}" \
    "${WINDOW_ASSIGNMENT_STRATEGY}" <<'REMOTE'
set -euo pipefail
RANK="$1"; NNODES="$2"; MASTER_ADDR="$3"; MASTER_PORT="$4"
DATASET="$5"; DATASET_ALIAS="$6"; NPARTS="$7"; EPOCHS="$8"; RUN_TAG="$9"
NCCL_SOCKET_IFNAME="${10}"; PEER_HOSTS="${11:-}"; SSH_KEY="${12}"
DATASET_DIR="${13}"; NEUTRON_ROOT="${14}"; PY="${15}"; LOG_ROOT="${16}"
WINDOW_EXTRA_RATIO="${17}"; WINDOW_RELATED_RATIO="${18}"; WINDOW_HUB_RATIO="${19}"
USE_PREPROCESS_CACHE="${20}"; REFRESH_PREPROCESS_CACHE="${21}"; SEQ_LEN="${22}"
WINDOW_ASSIGNMENT_STRATEGY="${23:-edge_balanced_step}"
[[ "${PEER_HOSTS}" == "-" ]] && PEER_HOSTS=""

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

cd "${NEUTRON_ROOT}"
ENTRY="main_sp_node_level_ppr.py"
if [[ "${NNODES}" -gt 1 ]]; then
  ENTRY="wrap_neutron_main.py"
fi

nohup env NEUTRONG_PEER_HOSTS="${PEER_HOSTS}" NEUTRONG_SSH_KEY="${SSH_KEY}" \
  "${PY}" -m torch.distributed.run \
  --nnodes="${NNODES}" --nproc_per_node=1 --node_rank="${RANK}" \
  --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" \
  "${ENTRY}" \
  --dataset "${DATASET}" --dataset_dir "${DATASET_DIR}/" \
  --model graphormer --attn_type sparse \
  --seq_len "${SEQ_LEN}" \
  --n_layers 4 --hidden_dim 64 --ffn_dim 64 --num_heads 8 \
  --epochs "${EPOCHS}" --use_cache 1 \
  --use_preprocess_cache "${USE_PREPROCESS_CACHE}" \
  --refresh_preprocess_cache "${REFRESH_PREPROCESS_CACHE}" \
  --n_parts "${NPARTS}" \
  --window_extra_node_ratio "${WINDOW_EXTRA_RATIO}" \
  --window_related_ratio "${WINDOW_RELATED_RATIO}" \
  --window_hub_ratio "${WINDOW_HUB_RATIO}" \
  --window_assignment_strategy "${WINDOW_ASSIGNMENT_STRATEGY}" \
  --ppr_backend appnp --ppr_topk 5 --ppr_alpha 0.85 \
  --ppr_num_iterations 10 --ppr_batch_size 8192 --ppr_iter_topk 5 \
  --distributed-backend nccl --distributed-timeout-minutes 120 \
  >"${LOG_FILE}" 2>&1 &

echo "$!" > "${PID_FILE}"
echo "started rank=${RANK} pid=$! log=${LOG_FILE}"
REMOTE
done

STATE_FILE="${LOG_ROOT}/state_${DATASET_ALIAS}_${NNODES}card.env"
mkdir -p "${LOG_ROOT}"
printf 'DATASET_ALIAS=%q\nNNODES=%q\nMASTER_ADDR=%q\nRUN_TAG=%q\nLOG_ROOT=%q\nNEUTRON_ROOT=%q\n' \
  "${DATASET_ALIAS}" "${NNODES}" "${MASTER_ADDR}" "${RUN_TAG}" "${LOG_ROOT}" "${NEUTRON_ROOT}" \
  > "${STATE_FILE}"

echo "Started: ${RUN_TAG}"
echo "Master log:"
echo "  tail -f ${LOG_ROOT}/${DATASET_ALIAS}/${NNODES}card/node0_${RUN_TAG}.log"
echo "Wait:"
echo "  bash ${CLUSTER_DIR}/wait.sh ${HOSTS_ARG} ${DATASET_ALIAS}"
