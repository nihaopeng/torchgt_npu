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
    echo "Usage: bash $0 <devices> [--epochs N]"
    echo "Example: bash $0 0,1,2"
    echo "         bash $0 0,1,2,3 --epochs 10"
    exit 1
fi
shift

EPOCHS=40
while [[ $# -gt 0 ]]; do
    case "$1" in
        --epochs)
            shift
            if [[ $# -eq 0 || "$1" == -* ]]; then
                echo "Error: --epochs requires a positive integer." >&2
                exit 1
            fi
            if ! [[ "$1" =~ ^[1-9][0-9]*$ ]]; then
                echo "Error: --epochs must be a positive integer, got: $1" >&2
                exit 1
            fi
            EPOCHS="$1"
            ;;
        *)
            echo "Error: unknown argument: $1" >&2
            echo "Usage: bash $0 <devices> [--epochs N]" >&2
            exit 1
            ;;
    esac
    shift
done

IFS=, read -r -a GPU_LIST <<< "$DEVICES"
if [ ${#GPU_LIST[@]} -lt 3 ]; then
    echo "Error: this scaling script needs at least 3 devices, got: ${DEVICES}" >&2
    exit 1
fi

DATASET="ogbn-arxiv"
DATASET_DIR=./dataset/
LOG_DIR=NeutronGT_logs/arxiv_gph_scaling_tmp
RUN_TAG=$(date +%Y%m%d_%H%M)

MODEL_ALIAS="GPH_Slim"
MODEL="graphormer"
N_LAYERS=4
HIDDEN_DIM=64
FFN_DIM=64
NUM_HEADS=8

ATTN_TYPE="sparse"
USE_CACHE=1
USE_PREPROCESS_CACHE=1
REFRESH_PREPROCESS_CACHE=0
SUBGRAPH_BUILDER="edge_scan"
NPARTS=16
WINDOW_AUG_STRATEGY="ours"
WINDOW_EXTRA_RATIO=0.30
WINDOW_RELATED_RATIO=0.15
WINDOW_HUB_RATIO=0.15

PPR_BACKEND="appnp"
PPR_TOPK=5
PPR_ALPHA=0.85
PPR_NUM_ITER=10
PPR_BATCH_SIZE=8192
PPR_ITER_TOPK=5
TIMEOUT=120

mkdir -p "${LOG_DIR}"

pick_free_port() {
    python -c 'import socket; s = socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()'
}

device_subset() {
    local count="$1"
    local subset="${GPU_LIST[0]}"
    local i
    for ((i = 1; i < count; i++)); do
        subset="${subset},${GPU_LIST[$i]}"
    done
    echo "${subset}"
}

parse_avg_epoch_time() {
    local log_file="$1"
    awk '
        /Training epoch time:/ {
            value = $3
            gsub("s", "", value)
            times[++n] = value + 0
        }
        END {
            start = (n > 1) ? 2 : 1
            for (i = start; i <= n; i++) {
                sum += times[i]
                count += 1
            }
            if (count > 0) {
                printf "%.6f", sum / count
            } else {
                printf "nan"
            }
        }
    ' "${log_file}"
}

SUMMARY_FILE="${LOG_DIR}/arxiv_${MODEL_ALIAS}_scaling_e${EPOCHS}_${RUN_TAG}.tsv"
echo -e "gpus\tdevices\tavg_epoch_time_s_skip_first\tspeedup_vs_1gpu\tlog" > "${SUMMARY_FILE}"

BASE_TIME=""
FAILED=0

for GPU_NUM in 1 2 3; do
    RUN_DEVICES=$(device_subset "${GPU_NUM}")
    MASTER_PORT=$(pick_free_port)
    LOG_FILE="${LOG_DIR}/${DATASET}_${MODEL_ALIAS}_${GPU_NUM}gpu_e${EPOCHS}_nparts${NPARTS}_${RUN_TAG}.log"

    echo "============================================================="
    echo "Arxiv GPH_Slim scaling temporary run"
    echo "GPUs=${GPU_NUM} CUDA_VISIBLE_DEVICES=${RUN_DEVICES}"
    echo "dataset=${DATASET} model=${MODEL_ALIAS} epochs=${EPOCHS} n_parts=${NPARTS}"
    echo "window_aug=${WINDOW_AUG_STRATEGY} extra=${WINDOW_EXTRA_RATIO} related=${WINDOW_RELATED_RATIO} hub=${WINDOW_HUB_RATIO}"
    echo "cache=${USE_CACHE} preprocess_cache=${USE_PREPROCESS_CACHE} subgraph_builder=${SUBGRAPH_BUILDER}"
    echo "master_port=${MASTER_PORT} log=${LOG_FILE}"
    echo "============================================================="

    CUDA_VISIBLE_DEVICES="${RUN_DEVICES}" torchrun \
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
        --subgraph_builder "${SUBGRAPH_BUILDER}" \
        --ppr_backend "${PPR_BACKEND}" \
        --ppr_topk "${PPR_TOPK}" \
        --ppr_alpha "${PPR_ALPHA}" \
        --ppr_num_iterations "${PPR_NUM_ITER}" \
        --ppr_batch_size "${PPR_BATCH_SIZE}" \
        --ppr_iter_topk "${PPR_ITER_TOPK}" \
        --distributed-backend nccl \
        --distributed-timeout-minutes "${TIMEOUT}" \
        > "${LOG_FILE}" 2>&1

    EXIT_CODE=$?
    if [ ${EXIT_CODE} -ne 0 ]; then
        echo "[${GPU_NUM} GPU] Failed (exit ${EXIT_CODE}). Check ${LOG_FILE}"
        echo -e "${GPU_NUM}\t${RUN_DEVICES}\tfailed\tfailed\t${LOG_FILE}" >> "${SUMMARY_FILE}"
        FAILED=1
        continue
    fi

    AVG_TIME=$(parse_avg_epoch_time "${LOG_FILE}")
    if [ -z "${BASE_TIME}" ] && [ "${AVG_TIME}" != "nan" ]; then
        BASE_TIME="${AVG_TIME}"
    fi
    SPEEDUP=$(awk -v base="${BASE_TIME}" -v current="${AVG_TIME}" 'BEGIN { if (base > 0 && current > 0) printf "%.3f", base / current; else printf "nan" }')
    echo -e "${GPU_NUM}\t${RUN_DEVICES}\t${AVG_TIME}\t${SPEEDUP}\t${LOG_FILE}" >> "${SUMMARY_FILE}"
    echo "[${GPU_NUM} GPU] avg_epoch_time_skip_first=${AVG_TIME}s speedup=${SPEEDUP}x"
done

echo "============================================================="
echo "Scaling summary: ${SUMMARY_FILE}"
cat "${SUMMARY_FILE}"
echo "============================================================="

if [ ${FAILED} -ne 0 ]; then
    exit 1
fi
