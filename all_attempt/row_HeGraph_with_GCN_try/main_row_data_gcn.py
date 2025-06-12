import torch
import numpy as np
import torch.nn as nn
from model_gcn import GCN
import pickle
import argparse
from sklearn.metrics import f1_score as sk_f1_score
from utils import init_seed, f1_score
import os

if __name__ == '__main__':
    init_seed(seed=777)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='DBLP',
                        help='Dataset')
    parser.add_argument('--epoch', type=int, default=200,
                        help='Training Epochs')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='hidden dimensions')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of GCN layers')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.001,
                        help='l2 reg')
    parser.add_argument('--runs', type=int, default=10,
                        help='number of runs')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--use_processed_edges', action='store_true',
                        help='使用处理后的边权重数据，否则使用原始边数据')
    parser.add_argument('--processed_epoch', type=int,
                        help='使用哪个epoch处理后的边权重数据（仅在use_processed_edges为True时有效）')
    parser.add_argument("--remove_self_loops", action='store_true', help="remove_self_loops")

    args = parser.parse_args()
    print(args)

    # 设置路径
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(base_dir, "data", args.dataset)
    processed_data_dir = os.path.join(base_dir, "metapath2vec_including_edge_weight", "process_data", "save_data")

    # 读取原始数据
    with open(os.path.join(data_dir, 'node_features.pkl'), 'rb') as f:
        node_features = pickle.load(f)
    with open(os.path.join(data_dir, 'labels.pkl'), 'rb') as f:
        labels = pickle.load(f)
    
    # 读取边数据
    if args.use_processed_edges:
        if args.processed_epoch is None:
            raise ValueError("使用处理后的边权重数据时必须指定processed_epoch参数")
        print(f"使用处理后的边权重数据 (Epoch {args.processed_epoch})")
        with open(os.path.join(processed_data_dir, f'Epoch{args.processed_epoch}_edges.pkl'), 'rb') as f:
            edges = pickle.load(f)
    else:
        print("使用原始边数据")
        with open(os.path.join(data_dir, 'edges.pkl'), 'rb') as f:
            edges = pickle.load(f)

    # 准备数据
    num_nodes = node_features.shape[0]
    node_features = torch.from_numpy(node_features).type(torch.cuda.FloatTensor)
    
    # 准备标签数据
    train_node = torch.from_numpy(np.array(labels[0])[:,0]).type(torch.cuda.LongTensor)
    train_target = torch.from_numpy(np.array(labels[0])[:,1]).type(torch.cuda.LongTensor)
    valid_node = torch.from_numpy(np.array(labels[1])[:,0]).type(torch.cuda.LongTensor)
    valid_target = torch.from_numpy(np.array(labels[1])[:,1]).type(torch.cuda.LongTensor)
    test_node = torch.from_numpy(np.array(labels[2])[:,0]).type(torch.cuda.LongTensor)
    test_target = torch.from_numpy(np.array(labels[2])[:,1]).type(torch.cuda.LongTensor)
    
    num_classes = np.max([torch.max(train_target).item(), torch.max(valid_target).item(), torch.max(test_target).item()])+1

    # 准备边数据
    edge_index = []
    edge_weight = []
    for matrix in edges:
        # 获取非零元素的位置和值
        indices = matrix.nonzero()
        values = matrix.data
        edge_index.append(torch.from_numpy(np.vstack((indices[0], indices[1]))).type(torch.cuda.LongTensor))
        edge_weight.append(torch.from_numpy(values).type(torch.cuda.FloatTensor))

    # 合并所有边
    total_edge_index = torch.cat(edge_index, dim=1)
    total_edge_weight = torch.cat(edge_weight)

    final_f1, final_micro_f1 = [], []
    
    for run in range(args.runs):
        print(f'\nRun {run + 1}:')
        
        # 初始化模型
        model = GCN(in_channels=node_features.shape[1],
                   hidden_channels=args.hidden_dim,
                   out_channels=num_classes,
                   num_layers=args.num_layers,
                   dropout=args.dropout,
                   args=args)
        
        model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
        best_val_loss = float('inf')
        best_test_loss = float('inf')
        best_train_loss = float('inf')
        best_train_f1, best_micro_train_f1 = 0, 0
        best_val_f1, best_micro_val_f1 = 0, 0
        best_test_f1, best_micro_test_f1 = 0, 0
        best_epoch = 0  # 添加记录最佳epoch的变量
        
        for epoch in range(args.epoch):
            model.train()
            optimizer.zero_grad()
            
            # 训练
            loss, y_train, _ = model(node_features, total_edge_index, total_edge_weight, train_node, train_target)
            train_f1 = torch.mean(f1_score(torch.argmax(y_train.detach(),dim=1), train_target, num_classes=num_classes)).cpu().numpy()
            sk_train_f1 = sk_f1_score(train_target.detach().cpu(), np.argmax(y_train.detach().cpu(), axis=1), average='micro')
            
            loss.backward()
            optimizer.step()
            
            # 验证
            model.eval()
            with torch.no_grad():
                val_loss, y_valid, _ = model(node_features, total_edge_index, total_edge_weight, valid_node, valid_target)
                val_f1 = torch.mean(f1_score(torch.argmax(y_valid,dim=1), valid_target, num_classes=num_classes)).cpu().numpy()
                sk_val_f1 = sk_f1_score(valid_target.detach().cpu(), np.argmax(y_valid.detach().cpu(), axis=1), average='micro')
                
                test_loss, y_test, _ = model(node_features, total_edge_index, total_edge_weight, test_node, test_target)
                test_f1 = torch.mean(f1_score(torch.argmax(y_test,dim=1), test_target, num_classes=num_classes)).cpu().numpy()
                sk_test_f1 = sk_f1_score(test_target.detach().cpu(), np.argmax(y_test.detach().cpu(), axis=1), average='micro')
            
            # 打印每个epoch的信息
            print(f'Epoch {epoch+1:03d}:')
            print(f'Train - Loss: {loss.detach().cpu().numpy():.4f}, Macro_F1: {train_f1:.4f}, Micro_F1: {sk_train_f1:.4f}')
            print(f'Valid - Loss: {val_loss.detach().cpu().numpy():.4f}, Macro_F1: {val_f1:.4f}, Micro_F1: {sk_val_f1:.4f}')
            print(f'Test  - Loss: {test_loss.detach().cpu().numpy():.4f}, Macro_F1: {test_f1:.4f}, Micro_F1: {sk_test_f1:.4f}')
            print('-' * 50)
            
            # 更新最佳结果
            if sk_val_f1 > best_micro_val_f1:
                best_epoch = epoch + 1  # 记录最佳epoch
                best_val_loss = val_loss.detach().cpu().numpy()
                best_test_loss = test_loss.detach().cpu().numpy()
                best_train_loss = loss.detach().cpu().numpy()
                best_train_f1 = train_f1
                best_val_f1 = val_f1
                best_test_f1 = test_f1
                best_micro_train_f1 = sk_train_f1
                best_micro_val_f1 = sk_val_f1
                best_micro_test_f1 = sk_test_f1
                print(f'*** New Best Validation F1: {sk_val_f1:.4f} at Epoch {best_epoch} ***')
                print('-' * 50)
        
        print('--------------------Best Result-------------------------')
        print(f'Best Epoch: {best_epoch}')
        print('Train - Loss: {:.4f}, Macro_F1: {:.4f}, Micro_F1: {:.4f}'.format(best_train_loss, best_train_f1, best_micro_train_f1))
        print('Valid - Loss: {:.4f}, Macro_F1: {:.4f}, Micro_F1: {:.4f}'.format(best_val_loss, best_val_f1, best_micro_val_f1))
        print('Test - Loss: {:.4f}, Macro_F1: {:.4f}, Micro_F1: {:.4f}'.format(best_test_loss, best_test_f1, best_micro_test_f1))
        
        final_f1.append(best_test_f1)
        final_micro_f1.append(best_micro_test_f1)

    print('--------------------Final Result-------------------------')
    print('Test - Macro_F1: {:.4f}+{:.4f}, Micro_F1:{:.4f}+{:.4f}'.format(np.mean(final_f1), np.std(final_f1), np.mean(final_micro_f1), np.std(final_micro_f1))) 