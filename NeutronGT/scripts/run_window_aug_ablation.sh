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
    echo "Usage: bash $0 <devices> [--arxiv|--amazon|--reddit|--products ...] [--preprocess_only]"
    echo "Example: bash $0 0,1,2,3"
    echo "         bash $0 0,1,2,3 --arxiv --products"
    exit 1
fi
shift

PREPROCESS_ONLY=0
SELECTED_DATASET_FLAGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --arxiv|--amazon|--reddit|--products) SELECTED_DATASET_FLAGS+=("$1") ;;
        --preprocess_only) PREPROCESS_ONLY=1 ;;
        *)
            echo "Usage: bash $0 <devices> [--arxiv|--amazon|--reddit|--products ...] [--preprocess_only]" >&2
            echo "Error: unknown argument: $1" >&2
            exit 1
            ;;
    esac
    shift
done

if [ ${#SELECTED_DATASET_FLAGS[@]} -eq 0 ]; then
    DATASET_FLAGS=(--arxiv --amazon --reddit --products)
else
    DATASET_FLAGS=("${SELECTED_DATASET_FLAGS[@]}")
fi

IFS=, read -r -a GPU_LIST <<< "$DEVICES"
GPU_NUM=${#GPU_LIST[@]}

DATASET_DIR=./dataset/
LOG_DIR=NeutronGT_logs/window_aug_ablation
RUN_TAG=$(date +%Y%m%d_%H%M)
EPOCHS=500
MODEL_ALIAS="GPH_Slim"
MODEL="graphormer"
N_LAYERS=4
HIDDEN_DIM=64
FFN_DIM=64
NUM_HEADS=8
ATTN_TYPE="sparse"
USE_CACHE=1
USE_PREPROCESS_CACHE=0
LOG_MEMORY_STATS=1
MEMORY_LOG_INTERVAL=1
TIMEOUT=120
PPR_BATCH_SIZE=8192
PPR_ITER_TOPK=5

mkdir -p "${LOG_DIR}"

pick_free_port() {
    python -c 'import socket; s = socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()'
}

resolve_dataset() {
    case "$1" in
        --arxiv) echo "ogbn-arxiv" ;;
        --amazon) echo "AmazonProducts" ;;
        --reddit) echo "reddit" ;;
        --products) echo "ogbn-products" ;;
        *) return 1 ;;
    esac
}

resolve_window_params() {
    local dataset="$1"
    if [ "$dataset" = "AmazonProducts" ]; then
        echo "192 0.25 0.125"
    elif [ "$dataset" = "ogbn-arxiv" ]; then
        echo "8 0.50 0.25"
    elif [ "$dataset" = "ogbn-products" ]; then
        echo "384 0.25 0.125"
    elif [ "$dataset" = "reddit" ]; then
        echo "32 0.30 0.15"
    else
        return 1
    fi
}

SUCCEEDED_RUNS=()
FAILED_RUNS=()

for DATASET_FLAG in "${DATASET_FLAGS[@]}"; do
    DATASET=$(resolve_dataset "${DATASET_FLAG}")
    read -r NPARTS FULL_EXTRA HALF_EXTRA <<< "$(resolve_window_params "${DATASET}")"

    for STAGE in hub_half hub_related; do
        if [ "$STAGE" = "hub_half" ]; then
            WINDOW_AUG_STRATEGY="hub"
            WINDOW_EXTRA_RATIO="${HALF_EXTRA}"
            WINDOW_RELATED_RATIO=0.0
            WINDOW_HUB_RATIO="${HALF_EXTRA}"
        else
            WINDOW_AUG_STRATEGY="ours"
            WINDOW_EXTRA_RATIO="${FULL_EXTRA}"
            WINDOW_RELATED_RATIO="${HALF_EXTRA}"
            WINDOW_HUB_RATIO="${HALF_EXTRA}"
        fi

        MODE_LABEL="train"
        if [ "$PREPROCESS_ONLY" -eq 1 ]; then
            MODE_LABEL="preprocess"
        fi

        LOG_FILE="${LOG_DIR}/${DATASET}_${MODEL_ALIAS}_${STAGE}_e${EPOCHS}_nparts${NPARTS}_${MODE_LABEL}_${RUN_TAG}.log"
        MASTER_PORT=$(pick_free_port)

        echo "============================================================="
        echo "Window augmentation cumulative ablation"
        echo "Dataset: ${DATASET}"
        echo "Model: ${MODEL_ALIAS}"
        echo "Stage: ${STAGE}"
        echo "strategy=${WINDOW_AUG_STRATEGY}"
        echo "n_parts=${NPARTS} epochs=${EPOCHS}"
        echo "window extra=${WINDOW_EXTRA_RATIO} related=${WINDOW_RELATED_RATIO} hub=${WINDOW_HUB_RATIO}"
        echo "cache=${USE_CACHE} preprocess_cache=${USE_PREPROCESS_CACHE} log_memory=${LOG_MEMORY_STATS} memory_interval=${MEMORY_LOG_INTERVAL}"
        echo "GPUs=${GPU_NUM} CUDA_VISIBLE_DEVICES=${DEVICES} master_port=${MASTER_PORT}"
        echo "Log: ${LOG_FILE}"
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
            --n_parts "${NPARTS}" \
            --window_aug_strategy "${WINDOW_AUG_STRATEGY}" \
            --window_extra_node_ratio "${WINDOW_EXTRA_RATIO}" \
            --window_related_ratio "${WINDOW_RELATED_RATIO}" \
            --window_hub_ratio "${WINDOW_HUB_RATIO}" \
            --log_memory_stats "${LOG_MEMORY_STATS}" \
            --memory_log_interval "${MEMORY_LOG_INTERVAL}" \
            --ppr_backend appnp \
            --ppr_topk 5 \
            --ppr_alpha 0.85 \
            --ppr_num_iterations 10 \
            --ppr_batch_size "${PPR_BATCH_SIZE}" \
            --ppr_iter_topk "${PPR_ITER_TOPK}" \
            --preprocess_only "${PREPROCESS_ONLY}" \
            --distributed-backend nccl \
            --distributed-timeout-minutes "${TIMEOUT}" \
            > "${LOG_FILE}" 2>&1

        EXIT_CODE=$?
        if [ ${EXIT_CODE} -ne 0 ]; then
            echo "[${DATASET} ${MODEL_ALIAS} ${STAGE}] Failed (exit ${EXIT_CODE}), continuing. Check ${LOG_FILE}"
            FAILED_RUNS+=("${DATASET} ${MODEL_ALIAS} ${STAGE} exit=${EXIT_CODE} log=${LOG_FILE}")
            continue
        fi
        SUCCEEDED_RUNS+=("${DATASET} ${MODEL_ALIAS} ${STAGE} log=${LOG_FILE}")
        echo "[${DATASET} ${MODEL_ALIAS} ${STAGE}] Done."
    done
done

echo "========== Run Summary =========="
if [ ${#SUCCEEDED_RUNS[@]} -gt 0 ]; then
    echo "Succeeded:"
    for RUN in "${SUCCEEDED_RUNS[@]}"; do
        echo "  ${RUN}"
    done
else
    echo "Succeeded: none"
fi

if [ ${#FAILED_RUNS[@]} -gt 0 ]; then
    echo "Failed:"
    for RUN in "${FAILED_RUNS[@]}"; do
        echo "  ${RUN}"
    done
    echo "Completed with failures."
    exit 1
fi

echo "All window augmentation ablations done."
