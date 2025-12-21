## multi npu
case $1 in
"metis")
CUDA_VISIBLE_DEVICES=$2 torchrun --nproc_per_node=1 --master_port 8082 main_sp_node_level_metis.py --dataset $3 --seq_len 6400 --n_layers 4 --hidden_dim 64 --ffn_dim 64 --num_heads 8 --epochs 1000 --model gt --distributed-backend 'nccl' --reorder
;;
"origin")
CUDA_VISIBLE_DEVICES=$2 torchrun --nproc_per_node=1 --master_port 8082 main_sp_node_level.py --dataset $3 --seq_len 6400 --n_layers 4 --hidden_dim 64 --ffn_dim 64 --num_heads 8 --epochs 1000 --model gt --distributed-backend 'nccl' --attn_type 'full' --reorder
;;
esac