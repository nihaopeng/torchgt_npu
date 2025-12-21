import torch
import torch.nn.functional as F
import numpy as np
from models.graphormer_dist_node_level import Graphormer
from models.gt_dist_node_level import GT
from utils.lr import PolynomialDecayLR
import argparse
import os
import time
import random
import pandas as pd
import torch.distributed as dist
from gt_sp.initialize import (
    initialize_distributed,
    sequence_parallel_is_initialized,
    get_sequence_parallel_group,
    get_sequence_parallel_world_size,
    get_sequence_parallel_src_rank,
    get_sequence_length_per_rank,
    set_global_token_indices,
    set_last_batch_global_token_indices,
)
from gt_sp.reducer import sync_params_and_buffers, Reducer
from gt_sp.evaluate import sparse_eval_gpu
from gt_sp.utils import random_split_idx, get_batch_reorder_blockize, check_conditions
from utils.parser_node_level import parser_add_main_args
from collections import deque
from core.metisPartition import PartitionTree
from utils.vis import pics_to_gif, vis_interface,high_attn_node_plot
import utils.vis as vis

def main():
    parser = argparse.ArgumentParser(description='TorchGT node-level training arguments.')
    parser_add_main_args(parser)
    args = parser.parse_args()
   
    # Initialize distributed 
    initialize_distributed(args)
    device = f'cuda:{torch.cuda.current_device()}' 
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    if args.rank == 0:
        os.makedirs(args.model_dir, exist_ok=True)
    
    # Dataset 
    feature = torch.load(args.dataset_dir + args.dataset + '/x.pt') # [N, x_dim]
    y = torch.load(args.dataset_dir + args.dataset + '/y.pt') # [N]
    edge_index = torch.load(args.dataset_dir + args.dataset + '/edge_index.pt') # [2, num_edges]
    N = feature.shape[0]

    if args.dataset == 'pokec':
        y = torch.clamp(y, min=0) 
    split_idx = random_split_idx(y, frac_train=0.6, frac_valid=0.2, frac_test=0.2, seed=args.seed)

    if args.rank == 0:
        print(args)
        print('Dataset load successfully')
        print(f"Train nodes: {split_idx['train'].shape[0]}, Val nodes: {split_idx['valid'].shape[0]}, Test nodes: {split_idx['test'].shape[0]}") 
        print(f"Training iters: {split_idx['train'].size(0) // args.seq_len + 1}, Val iters: {split_idx['valid'].size(0) // args.seq_len + 1}, Test iters: {split_idx['test'].size(0) // args.seq_len + 1}")
    
    # Broadcast train indexes to all ranks 
    seq_parallel_world_size = get_sequence_parallel_world_size() if sequence_parallel_is_initialized() else 1
    if seq_parallel_world_size > 1:
        src_rank = get_sequence_parallel_src_rank()
        group = get_sequence_parallel_group()

    train_idx = split_idx['train']
    
    if args.rank == 0:
        flatten_train_idx = train_idx.to('cuda')
    else:
        total_numel = train_idx.numel()
        flatten_train_idx = torch.empty(total_numel,
                                device=device,
                                dtype=torch.int64)
    # Broadcast
    if seq_parallel_world_size > 1:
        dist.broadcast(flatten_train_idx, src_rank, group=group)

    # Initialize global token indices
    seq_len_per_rank = get_sequence_length_per_rank()
    sub_real_seq_len = seq_len_per_rank + args.num_global_node
    global_token_indices = list(range(0, seq_parallel_world_size * sub_real_seq_len, sub_real_seq_len))

    # Last batch fix sequence length
    if flatten_train_idx.shape[0] % args.seq_len != 0:
        last_batch_node_num = flatten_train_idx.shape[0] % args.seq_len
        if last_batch_node_num % seq_parallel_world_size != 0:
            div = last_batch_node_num // seq_parallel_world_size
            last_batch_node_num = div * seq_parallel_world_size + (seq_parallel_world_size - 1)

        x_dummy_list = [t for t in torch.tensor_split(
            torch.zeros(last_batch_node_num, ), seq_parallel_world_size, dim=0)]
        sub_split_seq_lens = [t.shape[0] for t in x_dummy_list] # e.g., [14, 14, 14, 13]
        sub_real_seq_len = max(sub_split_seq_lens) + args.num_global_node
        global_token_indices_last_batch = list(range(0, seq_parallel_world_size * sub_real_seq_len, sub_real_seq_len))
    else:
        sub_split_seq_lens = None
        global_token_indices_last_batch = None
    set_global_token_indices(global_token_indices)
    set_last_batch_global_token_indices(global_token_indices_last_batch)

    if args.model == "graphormer":
        model = Graphormer(
            n_layers=args.n_layers,
            num_heads=args.num_heads,
            input_dim=feature.shape[1],
            hidden_dim=args.hidden_dim,
            output_dim=y.max().item()+1,
            attn_bias_dim=args.attn_bias_dim,
            dropout_rate=args.dropout_rate,
            input_dropout_rate=args.input_dropout_rate,
            attention_dropout_rate=args.attention_dropout_rate,
            ffn_dim=args.ffn_dim,
            num_global_node=args.num_global_node
        ).to(device)
    elif args.model == "gt":
        model = GT(
           n_layers=args.n_layers,
            num_heads=args.num_heads,
            input_dim=feature.shape[1],
            hidden_dim=args.hidden_dim,
            output_dim=y.max().item()+1,
            attn_bias_dim=args.attn_bias_dim,
            dropout_rate=args.dropout_rate,
            input_dropout_rate=args.input_dropout_rate,
            attention_dropout_rate=args.attention_dropout_rate,
            ffn_dim=args.ffn_dim,
            num_global_node=args.num_global_node
        ).to(device)
        
    if args.rank == 0:
        print('Model params:', sum(p.numel() for p in model.parameters()))

    # Sync params and buffers. Ensures all rank models start off at the same value
    if seq_parallel_world_size > 1:
        sync_params_and_buffers(model)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.peak_lr, weight_decay=args.weight_decay)
    lr_scheduler = PolynomialDecayLR(
            optimizer,
            warmup=args.warmup_updates,
            tot=args.epochs,
            lr=args.peak_lr,
            end_lr=args.end_lr,
            power=1.0)
    
    val_acc_list, test_acc_list, epoch_t_list = [], [], []
    best_model, best_val, best_test = None, float('-inf'), float('-inf')
    
    # 在训练开始前创建或清空CSV文件并写入表头
    import csv
    csv_file = "training_log.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Loss', 'Time', 'Train acc', 'Val acc', 'Test acc'])
    
    # =================== metis partition =========================
    partitionTree = PartitionTree(feature,edge_index,train_idx,4)
    root = partitionTree.partition_nodes_metis()
    # =============================================================

    for epoch in range(1, args.epochs + 1):
        model.to(device)
        model.train()
        
        loss_list, iter_t_list = [], []
        # =================== metis partition =========================
        metis_partition_parts,metis_partition_feature,metis_partition_nodes = partitionTree.get_final_partitions()
        # =============================================================
        
        scores = []
        for i in range(len(metis_partition_feature)):
            attn_type = "full"
            
            t1 = time.time()
            # out_i,score = model(x_i, attn_bias, edge_index_i, attn_type=attn_type)
            out_i,score_agg,score_spe = model(metis_partition_feature[i].to(device), None, None, attn_type=attn_type)
            # print(f"out_i:{out_i.shape},y[metis_partition_parts[i]]:{y[metis_partition_parts[i]].shape}")
            # print(f"edge_index shape:{edge_index.cpu().numpy().shape}")
            loss = F.nll_loss(out_i, y[metis_partition_parts[i]].to(device).long())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            
            # Sync all-reduce gradient 
            if seq_parallel_world_size > 1:
                for name, param in model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        param.grad.div_(get_sequence_parallel_world_size())
                        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, group=get_sequence_parallel_group())

            optimizer.step()  
            # torch.cuda.synchronize()   
            t2 = time.time() 
             
            iter_t_list.append(t2 - t1)
            if epoch % 20 ==0 and i==0:
                vis_interface(score_agg,score_spe,edge_index)
            if epoch == args.epochs-1:
                pics_to_gif(vis.score_hist_flist,"./vis/score_var.gif")
                vis.plot(vis.epochs,np.array(vis.score_neighbor_ratio_list).T,"./vis/高注意力邻居占比")
                vis.plot(vis.epochs,np.array(vis.score_relativity_ratio_list).T,"./vis/高注意力相对应比例")
                high_attn_node_plot()
                vis.acc_plot(vis.epochs,[vis.train_acc,vis.test_acc,vis.val_acc],["train_acc","test_acc","val_acc"])
     
        loss_list.append(loss.item()) 
        lr_scheduler.step()
        
        if (epoch+1) % 20 == 0:
            print(f"score:{scores[0]}")
            partitionTree.dynamic_window_build(scores,metis_partition_nodes)
        
        csv_content = []
        
        if epoch > 4 and args.rank == 0:  
            epoch_t_list.append(np.sum(iter_t_list))
            print("------------------------------------------------------------------------------------")
            print("Epoch: {:03d}, Loss: {:.4f}, Epoch Time: {:.3f}s".format(epoch, np.mean(loss_list), np.mean(epoch_t_list)))
            # 在每个epoch结束后写入数据
            csv_content.append(epoch)
            csv_content.append(np.mean(loss_list))
            csv_content.append(np.mean(epoch_t_list))
            print("------------------------------------------------------------------------------------")

        if args.rank == 0 and epoch % 5 == 0:   
            t4 = time.time()
            train_acc = sparse_eval_gpu(args, model, feature, y, split_idx['train'], None, edge_index, device)
            val_acc = sparse_eval_gpu(args, model, feature, y, split_idx['valid'], None, edge_index, device)
            test_acc = sparse_eval_gpu(args, model, feature, y, split_idx['test'], None, edge_index, device)
            if epoch % 20 ==0 and i==0:
                vis.train_acc.append(train_acc)
                vis.val_acc.append(val_acc)
                vis.test_acc.append(test_acc)
            
            t5 = time.time()
            print("------------------------------------------------------------------------------------")
            print(f'Eval time {t5-t4}s')
            print("Epoch: {:03d}, Loss: {:4f}, Train acc: {:.2%}, Val acc: {:.2%}, Test acc: {:.2%}, Epoch Time: {:.3f}s".format(
                epoch, np.mean(loss_list), train_acc, val_acc, test_acc, np.mean(epoch_t_list)))
            csv_content.append(train_acc)
            csv_content.append(val_acc)
            csv_content.append(test_acc)
            print("------------------------------------------------------------------------------------")
        
            if val_acc > best_val:
                best_val = val_acc
                if args.save_model:
                    torch.save(model.state_dict(), args.model_dir + f'{args.dataset}.pkl')
            
            if test_acc > best_test:
                best_test = test_acc
            
            val_acc_list.append(val_acc)
            test_acc_list.append(test_acc)
            
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if csv_content:
                writer.writerow(csv_content)
    
    if args.rank == 0:
        print("Best validation accuracy: {:.2%}, test accuracy: {:.2%}".format(best_val, best_test))

        if not os.path.exists(f'./exps/{args.dataset}'): 
            os.makedirs(f'./exps/{args.dataset}')
        
        if args.attn_type != "hybrid":
            if args.reorder:
                np.save(f'./exps/{args.dataset}/{args.model}{args.hidden_dim}_{str(args.attn_type)}_reorder_s{args.seq_len}_e{args.epochs}_sp{args.world_size}_test-fp16', np.array(test_acc_list))
                # np.save('./exps/' + args.dataset + '/tt-sparse_bias_val_e' + str(args.epochs), np.array(val_acc_list))
                np.save(f'./exps/{args.dataset}/{args.model}{args.hidden_dim}_{str(args.attn_type)}_reorder_s{args.seq_len}_e{args.epochs}_sp{args.world_size}_loss-fp16', np.array(loss_list))
            else:
                np.save(f'./exps/{args.dataset}/{args.model}{args.hidden_dim}_{str(args.attn_type)}_s{args.seq_len}_e{args.epochs}_sp{args.world_size}_test', np.array(test_acc_list))
                # np.save('./exps/' + args.dataset + '/tt-sparse_bias_val_e' + str(args.epochs), np.array(val_acc_list))
                np.save(f'./exps/{args.dataset}/{args.model}{args.hidden_dim}_{str(args.attn_type)}_s{args.seq_len}_e{args.epochs}_sp{args.world_size}_loss', np.array(loss_list))
        else:
            np.save(f'./exps/{args.dataset}/{args.model}{args.hidden_dim}_{str(args.attn_type)}_{args.switch_freq}_s{args.seq_len}_e{args.epochs}_sp{args.world_size}_test', np.array(test_acc_list))
            # np.save('./exps/' + args.dataset + '/tt-sparse_bias_val_e' + str(args.epochs), np.array(val_acc_list))
            np.save(f'./exps/{args.dataset}/{args.model}{args.hidden_dim}_{str(args.attn_type)}_{args.switch_freq}_s{args.seq_len}_e{args.epochs}_sp{args.world_size}_loss', np.array(loss_list))


if __name__ == "__main__":
    main()
