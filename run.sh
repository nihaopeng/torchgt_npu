## multi npu
case $1 in
"metis")
CUDA_VISIBLE_DEVICES=$2 torchrun --nproc_per_node=1 --master_port 8082 main_sp_node_level_metis.py --dataset cora --seq_len 6400 --n_layers 4 --hidden_dim 64 --ffn_dim 64 --num_heads 8 --epochs 1000 --model gt --distributed-backend 'nccl' --reorder
;;
"origin")
CUDA_VISIBLE_DEVICES=$2 torchrun --nproc_per_node=1 --master_port 8081 main_sp_node_level.py --dataset cora --seq_len 6400 --n_layers 4 --hidden_dim 64 --ffn_dim 64 --num_heads 8 --epochs 1000 --model gt --distributed-backend 'nccl' --attn_type 'full' --reorder
;;
"attention_pruning")
    # 剪枝实验分支
    CUDA_VISIBLE_DEVICES=$2 torchrun --nproc_per_node=1 --master_port 8083 main_sp_node_level.py \
        --dataset cora --seq_len 6400 --n_layers 4 --hidden_dim 64 --ffn_dim 64 --num_heads 8 \
        --epochs 1000 --model gt --distributed-backend 'nccl' --attn_type 'full' --reorder \
        --enable_attention_pruning \
        --attention_pruning_ratio $3 \
;;
"neighbor_pruning")
    # 剪枝实验分支
    CUDA_VISIBLE_DEVICES=$2 torchrun --nproc_per_node=1 --master_port 8083 main_sp_node_level.py \
        --dataset cora --seq_len 6400 --n_layers 4 --hidden_dim 64 --ffn_dim 64 --num_heads 8 \
        --epochs 1000 --model gt --distributed-backend 'nccl' --attn_type 'full' --reorder \
        --enable_neighbor_pruning \
        --neighbor_pruning_ratio $3 \
;;

esac