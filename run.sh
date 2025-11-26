## multi npu
CUDA_VISIBLE_DEVICES=1,2 torchrun --nproc_per_node=1 --master_port 8081 main_sp_node_level.py --dataset ogbn-arxiv --seq_len 6400 --n_layers 4 --hidden_dim 64 --ffn_dim 64 --num_heads 8 --epochs 1000 --model gt --distributed-backend 'nccl' --reorder
