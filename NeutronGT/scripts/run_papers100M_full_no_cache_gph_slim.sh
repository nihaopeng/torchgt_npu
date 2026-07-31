#!/usr/bin/env bash

set -u
set -o pipefail

cd "$(dirname "$0")/.."

export CUDA_HOME=/usr/local/cuda-12.1
export CUDA_PATH=/usr/local/cuda-12.1
export PATH=$CUDA_HOME/bin:${PATH:-}
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}

DEVICES=${1-}
if [ -z "$DEVICES" ] || [[ "$DEVICES" == -* ]]; then
    echo "Usage: bash $0 <devices>"
    echo "Example: bash $0 0,1,2,3"
    exit 1
fi

IFS=, read -r -a GPU_LIST <<< "$DEVICES"
GPU_NUM=${#GPU_LIST[@]}

DATASET="ogbn-papers100M"
DATASET_DIR=./dataset/
LOG_DIR=NeutronGT_logs/main_ablation
RUN_TAG=$(date +%Y%m%d_%H%M)

MODEL_ALIAS="GPH_Slim"
MODEL="graphormer"
N_LAYERS=4
HIDDEN_DIM=64
FFN_DIM=64
NUM_HEADS=8
EPOCHS=40

ATTN_TYPE="full"
USE_CACHE=0
USE_PREPROCESS_CACHE=0
REFRESH_PREPROCESS_CACHE=0
NPARTS=16384
WINDOW_AUG_STRATEGY="ours"
WINDOW_EXTRA_RATIO=0.05
WINDOW_RELATED_RATIO=0.04
WINDOW_HUB_RATIO=0.01

PPR_BACKEND="appnp"
PPR_TOPK=5
PPR_ALPHA=0.85
PPR_NUM_ITER=10
PPR_BATCH_SIZE=2048
PPR_ITER_TOPK=16
TIMEOUT=640

LOG_MEMORY_STATS=1
MEMORY_LOG_INTERVAL=1

mkdir -p "${LOG_DIR}"

pick_free_port() {
    python -c 'import socket; s = socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()'
}

MASTER_PORT=$(pick_free_port)
LOG_FILE="${LOG_DIR}/${DATASET}_${MODEL_ALIAS}_full_no_cache_e${EPOCHS}_nparts${NPARTS}_${RUN_TAG}.log"

echo "============================================================="
echo "  NeutronGT paper100M full attention no-cache test"
echo "============================================================="
echo "  dataset=${DATASET}"
echo "  model=${MODEL_ALIAS} (${MODEL})"
echo "  layers=${N_LAYERS} hidden=${HIDDEN_DIM} ffn=${FFN_DIM} heads=${NUM_HEADS}"
echo "  epochs=${EPOCHS} n_parts=${NPARTS}"
echo "  attn_type=${ATTN_TYPE} use_cache=${USE_CACHE} use_preprocess_cache=${USE_PREPROCESS_CACHE}"
echo "  window_aug=${WINDOW_AUG_STRATEGY} extra=${WINDOW_EXTRA_RATIO} related=${WINDOW_RELATED_RATIO} hub=${WINDOW_HUB_RATIO}"
echo "  ppr_backend=${PPR_BACKEND} ppr_topk=${PPR_TOPK} ppr_batch=${PPR_BATCH_SIZE} ppr_iter_topk=${PPR_ITER_TOPK}"
echo "  log_memory_stats=${LOG_MEMORY_STATS} memory_log_interval=${MEMORY_LOG_INTERVAL}"
echo "  GPUs=${GPU_NUM} CUDA_VISIBLE_DEVICES=${DEVICES} master_port=${MASTER_PORT} timeout=${TIMEOUT}m"
echo "  log=${LOG_FILE}"
echo "============================================================="

CUDA_VISIBLE_DEVICES="${DEVICES}" torchrun \
    --nproc_per_node="${GPU_NUM}" \
    --master_port="${MASTER_PORT}" \
    main_sp_node_level_ppr.py \
    --dataset "${DATASET}" \
    --dataset_dir "${DATASET_DIR}" \
    --model "${MODEL}" \
    --attn_type "${ATTN_TYPE}" \
    --n_layers "${N_LAYERS}" \
    --hidden_dim "${HIDDEN_DIM}" \
    --ffn_dim "${FFN_DIM}" \
    --num_heads "${NUM_HEADS}" \
    --epochs "${EPOCHS}" \
    --use_cache "${USE_CACHE}" \
    --use_preprocess_cache "${USE_PREPROCESS_CACHE}" \
    --refresh_preprocess_cache "${REFRESH_PREPROCESS_CACHE}" \
    --n_parts "${NPARTS}" \
    --window_aug_strategy "${WINDOW_AUG_STRATEGY}" \
    --window_extra_node_ratio "${WINDOW_EXTRA_RATIO}" \
    --window_related_ratio "${WINDOW_RELATED_RATIO}" \
    --window_hub_ratio "${WINDOW_HUB_RATIO}" \
    --ppr_backend "${PPR_BACKEND}" \
    --ppr_topk "${PPR_TOPK}" \
    --ppr_alpha "${PPR_ALPHA}" \
    --ppr_num_iterations "${PPR_NUM_ITER}" \
    --ppr_batch_size "${PPR_BATCH_SIZE}" \
    --ppr_iter_topk "${PPR_ITER_TOPK}" \
    --log_memory_stats "${LOG_MEMORY_STATS}" \
    --memory_log_interval "${MEMORY_LOG_INTERVAL}" \
    --distributed-backend nccl \
    --distributed-timeout-minutes "${TIMEOUT}" \
    > "${LOG_FILE}" 2>&1

EXIT_CODE=$?
if [ ${EXIT_CODE} -ne 0 ]; then
    echo "Status: Failed (exit ${EXIT_CODE}). Check ${LOG_FILE}"
    exit ${EXIT_CODE}
fi

echo "Status: Success. Log saved to ${LOG_FILE}"
