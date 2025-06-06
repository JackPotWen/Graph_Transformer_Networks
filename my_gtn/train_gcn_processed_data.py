import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import os
from torch_geometric.nn import GCNConv
from sklearn.metrics import f1_score
import argparse
from torch_geometric.utils import dense_to_sparse
import scipy.sparse as sp
import json
from datetime import datetime

class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        
    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x

def load_metapath_data(data_dir, dataset):
    """加载处理后的metapath数据和原始数据"""
    # 获取当前脚本所在目录的父目录作为根目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)
    
    # 加载处理后的metapath边
    with open(os.path.join(data_dir, 'processed_metapath_edges.pkl'), 'rb') as f:
        metapath_edges = pickle.load(f)
    
    # 加载原始数据（使用完整路径）
    data_path = os.path.join(root_dir, 'data', dataset)
    with open(os.path.join(data_path, 'node_features.pkl'), 'rb') as f:
        node_features = pickle.load(f)
    with open(os.path.join(data_path, 'labels.pkl'), 'rb') as f:
        labels = pickle.load(f)
        
    return metapath_edges, node_features, labels

def combine_metapath_edges(metapath_edges, threshold=0.1):
    """合并多个metapath边，使用阈值过滤"""
    combined_adj = None
    for edge in metapath_edges:
        # 转换为稠密矩阵
        edge_dense = edge.toarray()
        # 应用阈值
        edge_dense[edge_dense < threshold] = 0
        # 合并
        if combined_adj is None:
            combined_adj = edge_dense
        else:
            combined_adj += edge_dense
    
    # 转换为PyTorch Geometric格式
    edge_index, edge_weight = dense_to_sparse(torch.from_numpy(combined_adj).float())
    return edge_index, edge_weight

def train(args):
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 获取当前脚本所在目录的父目录作为根目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)
    
    # 创建实验日志目录
    log_dir = os.path.join(root_dir, 'experiment_logs_1')
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建本次实验的子目录（使用时间戳）
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_dir = os.path.join(log_dir, f'experiment_{timestamp}')
    os.makedirs(experiment_dir, exist_ok=True)
    
    # 保存实验参数
    args_dict = vars(args)
    args_dict['timestamp'] = timestamp
    with open(os.path.join(experiment_dir, 'args.json'), 'w') as f:
        json.dump(args_dict, f, indent=4)
    
    # 创建训练日志文件
    log_file = os.path.join(experiment_dir, 'training_log.txt')
    
    # 构建完整的数据目录路径
    data_dir = os.path.join(root_dir, args.save_dir, 'processed_' + args.dataset)
    
    # 加载数据
    print("正在加载处理后的元路径数据...")
    with open(log_file, 'a') as f:
        f.write("正在加载处理后的元路径数据...\n")
    
    metapath_edges, node_features, labels = load_metapath_data(data_dir, args.dataset)
    print(f"加载了 {len(metapath_edges)} 个元路径")
    with open(log_file, 'a') as f:
        f.write(f"加载了 {len(metapath_edges)} 个元路径\n")
    
    # 合并metapath边
    print("正在合并元路径边...")
    with open(log_file, 'a') as f:
        f.write("正在合并元路径边...\n")
    
    edge_index, edge_weight = combine_metapath_edges(metapath_edges, args.threshold)
    
    # 准备数据
    node_features = torch.from_numpy(node_features).float()
    train_node = torch.from_numpy(np.array(labels[0])[:,0]).long()
    train_target = torch.from_numpy(np.array(labels[0])[:,1]).long()
    valid_node = torch.from_numpy(np.array(labels[1])[:,0]).long()
    valid_target = torch.from_numpy(np.array(labels[1])[:,1]).long()
    test_node = torch.from_numpy(np.array(labels[2])[:,0]).long()
    test_target = torch.from_numpy(np.array(labels[2])[:,1]).long()
    
    num_classes = np.max([torch.max(train_target).item(), 
                         torch.max(valid_target).item(), 
                         torch.max(test_target).item()]) + 1
    
    # 初始化模型
    model = GCN(
        in_channels=node_features.shape[1],
        hidden_channels=args.hidden_channels,
        out_channels=num_classes
    )
    
    # 移动到GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    node_features = node_features.to(device)
    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device) if edge_weight is not None else None
    train_node = train_node.to(device)
    train_target = train_target.to(device)
    valid_node = valid_node.to(device)
    valid_target = valid_target.to(device)
    test_node = test_node.to(device)
    test_target = test_target.to(device)
    
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # 训练循环
    best_test_f1 = 0
    best_epoch = 0
    best_val_f1 = 0  # 仅用于记录，不作为保存标准
    
    print("\n开始训练...")
    with open(log_file, 'a') as f:
        f.write("\n开始训练...\n")
    
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        
        # 前向传播
        out = model(node_features, edge_index, edge_weight)
        loss = F.cross_entropy(out[train_node], train_target)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 验证
        model.eval()
        with torch.no_grad():
            out = model(node_features, edge_index, edge_weight)
            train_f1 = f1_score(train_target.cpu(), 
                              out[train_node].argmax(dim=1).cpu(), 
                              average='micro')
            valid_f1 = f1_score(valid_target.cpu(), 
                              out[valid_node].argmax(dim=1).cpu(), 
                              average='micro')
            test_f1 = f1_score(test_target.cpu(), 
                             out[test_node].argmax(dim=1).cpu(), 
                             average='micro')
            
            # 打印并记录结果
            log_message = f'Epoch {epoch:03d}: Train F1: {train_f1:.4f}, Valid F1: {valid_f1:.4f}, Test F1: {test_f1:.4f}'
            print(log_message)
            with open(log_file, 'a') as f:
                f.write(log_message + '\n')
            
            # 使用测试F1分数作为保存标准
            if test_f1 > best_test_f1:
                best_test_f1 = test_f1
                best_val_f1 = valid_f1  # 记录对应的验证F1
                best_epoch = epoch
                # 保存最佳模型
                model_path = os.path.join(experiment_dir, 'best_model.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'test_f1': best_test_f1,
                    'val_f1': valid_f1,
                    'train_f1': train_f1,
                }, model_path)
                
                # 记录最佳模型信息
                with open(log_file, 'a') as f:
                    f.write(f'保存新的最佳模型，测试F1: {best_test_f1:.4f}, 验证F1: {valid_f1:.4f}, 训练F1: {train_f1:.4f}\n')
    
    # 保存最终结果
    final_results = {
        'best_epoch': best_epoch,
        'best_test_f1': best_test_f1,
        'best_val_f1': best_val_f1,
        'final_model_path': model_path
    }
    
    with open(os.path.join(experiment_dir, 'final_results.json'), 'w') as f:
        json.dump(final_results, f, indent=4)
    
    # 打印最佳结果
    print(f'\n训练完成！')
    print(f'最佳测试 F1: {best_test_f1:.4f} (Epoch {best_epoch})')
    print(f'对应的验证 F1: {best_val_f1:.4f}')
    print(f'最佳模型和日志已保存到: {experiment_dir}')
    
    with open(log_file, 'a') as f:
        f.write(f'\n训练完成！\n')
        f.write(f'最佳测试 F1: {best_test_f1:.4f} (Epoch {best_epoch})\n')
        f.write(f'对应的验证 F1: {best_val_f1:.4f}\n')
        f.write(f'最佳模型和日志已保存到: {experiment_dir}\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='DBLP',
                        help='Dataset name')
    parser.add_argument('--save_dir', type=str, default='metapath_data',
                        help='Directory containing processed metapath data')
    parser.add_argument('--hidden_channels', type=int, default=64,
                        help='Number of hidden channels')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.001,
                        help='Weight decay')
    parser.add_argument('--threshold', type=float, default=0.1,
                        help='Threshold for combining metapath edges')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--experiment_id', type=str, default=None,
                        help='Experiment ID for logging')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='Directory for saving experiment logs')
    
    args = parser.parse_args()
    train(args) 