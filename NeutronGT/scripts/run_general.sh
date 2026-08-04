#!/usr/bin/env bash

set -u
set -o pipefail

cd "$(dirname "$0")/.."

export CUDA_HOME=/usr/local/cuda-12.1
export CUDA_PATH=/home/pengyt/softwares/miniconda/envs/gt/
export PATH=$CUDA_HOME/bin:${PATH:-}
export LD_LIBRARY_PATH=$CUDA_PATH/lib:${LD_LIBRARY_PATH:-}

# ====================== Defaults ======================
DATASET="ogbn-arxiv"
MODEL="gt_sw"
N_LAYERS=4
HIDDEN_DIM=128
FFN_DIM=128
NUM_HEADS=8
ATTN_TYPE="sparse"
EPOCHS=500
NPARTS=16
RELATED_TOPK=8
PPR_BACKEND="appnp"
PPR_TOPK=5
PPR_ALPHA=0.85
WINDOW_ASSIGN="edge_balanced_step"
PPR_NUM_ITER=10
PPR_BATCH_SIZE=8192
PPR_ITER_TOPK=5
SEQ_LEN=256000
USE_CACHE=1
USE_PREPROCESS_CACHE=0
DATASET_DIR="./dataset/"
LOG_DIR="NeutronGT_logs"
RUN_TAG=$(date +%Y%m%d_%H%M)
MASTER_PORT=$((8000 + RANDOM % 1000))

# ====================== Help ======================
usage() {
    cat <<EOF
Usage: bash $0 <devices> [options]

Required:
  <devices>              CUDA_VISIBLE_DEVICES, e.g. 0 or 0,1,2,3

Dataset shortcuts (mutually exclusive):
  --arxiv                ogbn-arxiv
  --amazon               AmazonProducts
  --reddit               reddit
  --products             ogbn-products
  --dataset NAME         Any dataset name in dataset_dir

Model (override individual params):
  --n_layers N           (default: $N_LAYERS)
  --hidden_dim D         (default: $HIDDEN_DIM)
  --ffn_dim D            (default: $FFN_DIM)
  --num_heads H          (default: $NUM_HEADS)
  --attn_type TYPE       full | sparse (default: $ATTN_TYPE)
  --epochs E             (default: $EPOCHS)
  --seq_len L            max sequence length (default: $SEQ_LEN)

Partition:
  --n_parts N            number of partitions (default: $NPARTS)
  --related_topk R       halo extension top-k% (default: $RELATED_TOPK)
  --ppr_topk K           PPR neighbors per node (default: $PPR_TOPK)
  --ppr_alpha A          PPR teleport prob (default: $PPR_ALPHA)
  --window_assign S      window-to-rank assignment: edge_balanced_step | round_robin (default: $WINDOW_ASSIGN)

PPR backend:
  --ppr_backend NAME     appnp | torch_geometric (default: $PPR_BACKEND)
  --ppr_num_iter I       APPNP iterations (default: $PPR_NUM_ITER)
  --ppr_batch_size B     APPNP batch size (default: $PPR_BATCH_SIZE)
  --ppr_iter_topk K      APPNP iter top-k pruning (default: $PPR_ITER_TOPK)

Cache:
  --use_cache 0|1        (default: $USE_CACHE)
  --refresh_cache        equivalent to --use_preprocess_cache 0

Other:
  --dataset_dir DIR      (default: $DATASET_DIR)
  --log_dir DIR          (default: $LOG_DIR)

Examples:
  bash $0 0 --arxiv
  bash $0 0,1,2,3 --arxiv --n_parts 32 --related_topk 10
  bash $0 0 --amazon --n_parts 64 --epochs 200
  bash $0 0,1 --dataset cora --n_parts 4 --attn_type full --epochs 300
EOF
    exit 1
}

# ====================== Parse Args ======================
DEVICES=${1-}
if [ -z "$DEVICES" ] || [[ "$DEVICES" == -* ]]; then
    usage
fi
shift

while [[ $# -gt 0 ]]; do
    case "$1" in
        --arxiv)       DATASET="ogbn-arxiv" ;;
        --amazon)      DATASET="AmazonProducts" ;;
        --reddit)      DATASET="reddit" ;;
        --products)    DATASET="ogbn-products" ;;
        --dataset)     DATASET="$2"; shift ;;
        --n_layers)    N_LAYERS="$2"; shift ;;
        --hidden_dim)  HIDDEN_DIM="$2"; shift ;;
        --ffn_dim)     FFN_DIM="$2"; shift ;;
        --num_heads)   NUM_HEADS="$2"; shift ;;
        --attn_type)   ATTN_TYPE="$2"; shift ;;
        --epochs)      EPOCHS="$2"; shift ;;
        --seq_len)     SEQ_LEN="$2"; shift ;;
        --n_parts)     NPARTS="$2"; shift ;;
        --related_topk) RELATED_TOPK="$2"; shift ;;
        --ppr_topk)    PPR_TOPK="$2"; shift ;;
        --ppr_alpha)   PPR_ALPHA="$2"; shift ;;
        --window_assign) WINDOW_ASSIGN="$2"; shift ;;
        --ppr_backend) PPR_BACKEND="$2"; shift ;;
        --ppr_num_iter) PPR_NUM_ITER="$2"; shift ;;
        --ppr_batch_size) PPR_BATCH_SIZE="$2"; shift ;;
        --ppr_iter_topk) PPR_ITER_TOPK="$2"; shift ;;
        --use_cache)   USE_CACHE="$2"; shift ;;
        --refresh_cache) USE_PREPROCESS_CACHE=0 ;;
        --dataset_dir) DATASET_DIR="$2"; shift ;;
        --log_dir)     LOG_DIR="$2"; shift ;;
        -h|--help)     usage ;;
        *)
            echo "Error: unknown argument: $1" >&2
            usage
            ;;
    esac
    shift
done

# ====================== Execute ======================

IFS=, read -r -a GPU_LIST <<< "$DEVICES"
GPU_NUM=${#GPU_LIST[@]}

mkdir -p "${LOG_DIR}"

LOG_FILE="${LOG_DIR}/${DATASET}_general_nparts${NPARTS}_rtopk${RELATED_TOPK}_${RUN_TAG}.log"

echo "-------------------------------------------------------------"
echo "Dataset:       ${DATASET}"
echo "GPUs:          ${GPU_NUM} (CUDA_VISIBLE_DEVICES=${DEVICES})"
echo "Model:         ${MODEL}, layers=${N_LAYERS}, dim=${HIDDEN_DIM}/${FFN_DIM}, heads=${NUM_HEADS}"
echo "Attention:     ${ATTN_TYPE}"
echo "Epochs:        ${EPOCHS}"
echo "Partitions:    ${NPARTS}, related_topk=${RELATED_TOPK}%"
echo "PPR:           backend=${PPR_BACKEND}, topk=${PPR_TOPK}, alpha=${PPR_ALPHA}"
echo "Log:           ${LOG_FILE}"
echo "-------------------------------------------------------------"

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
    --seq_len "${SEQ_LEN}" \
    --use_cache "${USE_CACHE}" \
    --use_preprocess_cache "${USE_PREPROCESS_CACHE}" \
    --n_parts "${NPARTS}" \
    --related_nodes_topk_rate "${RELATED_TOPK}" \
    --ppr_backend "${PPR_BACKEND}" \
    --ppr_topk "${PPR_TOPK}" \
    --ppr_alpha "${PPR_ALPHA}" \
    --ppr_num_iterations "${PPR_NUM_ITER}" \
    --ppr_batch_size "${PPR_BATCH_SIZE}" \
    --ppr_iter_topk "${PPR_ITER_TOPK}" \
    --window_assignment_strategy "${WINDOW_ASSIGN}" \
    --distributed-backend nccl \
    --distributed-timeout-minutes 120 \
    > "${LOG_FILE}" 2>&1

if [ $? -eq 0 ]; then
    echo "Status: Success"
else
    echo "Status: Failed. Check ${LOG_FILE}"
fi
