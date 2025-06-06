import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import pickle
import os
import argparse
from torch_geometric.utils import add_self_loops
from sklearn.metrics import f1_score
from model import GTN
import scipy.sparse as sp

def save_metapath_graph(A, Ws, num_nodes, save_dir):
    """保存metapath图结构"""
    # 构建新的邻接矩阵
    new_edges = []
    for W in Ws:
        # 将权重矩阵转换为numpy数组
        w_np = W.detach().cpu().numpy()
        # 对每个通道构建新的边
        for i in range(w_np.shape[0]):
            for j in range(w_np.shape[1]):
                if w_np[i,j] > 0.1:  # 只保留权重较大的边
                    # 获取原始边
                    edge_i, value_i = A[i]
                    edge_j, value_j = A[j]
                    # 构建新的边
                    edge_i = edge_i.cpu().numpy()
                    edge_j = edge_j.cpu().numpy()
                    value_i = value_i.cpu().numpy()
                    value_j = value_j.cpu().numpy()
                    
                    # 构建稀疏矩阵
                    adj_i = sp.coo_matrix((value_i, (edge_i[0], edge_i[1])), shape=(num_nodes, num_nodes))
                    adj_j = sp.coo_matrix((value_j, (edge_j[0], edge_j[1])), shape=(num_nodes, num_nodes))
                    
                    # 矩阵乘法得到新的边
                    new_adj = adj_i.dot(adj_j)
                    new_edges.append(new_adj)
    
    # 保存新的边
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'metapath_edges.pkl'), 'wb') as f:
        pickle.dump(new_edges, f)
    
    # 保存权重矩阵
    with open(os.path.join(save_dir, 'metapath_weights.pkl'), 'wb') as f:
        pickle.dump([w.detach().cpu().numpy() for w in Ws], f)

def train(args):
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 获取当前文件所在目录的父目录（项目根目录）
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)
    
    # 加载数据
    data_dir = os.path.join(root_dir, 'data', args.dataset)
    with open(os.path.join(data_dir, 'node_features.pkl'), 'rb') as f:
        node_features = pickle.load(f)
    with open(os.path.join(data_dir, 'edges.pkl'), 'rb') as f:
        edges = pickle.load(f)
    with open(os.path.join(data_dir, 'labels.pkl'), 'rb') as f:
        labels = pickle.load(f)
    
    # 构建邻接矩阵
    num_nodes = edges[0].shape[0]
    A = []
    for i, edge in enumerate(edges):
        edge_tmp = torch.from_numpy(np.vstack((edge.nonzero()[1], edge.nonzero()[0]))).type(torch.cuda.LongTensor)
        value_tmp = torch.ones(edge_tmp.shape[1]).type(torch.cuda.FloatTensor)
        A.append((edge_tmp, value_tmp))
    
    # 添加自环
    edge_tmp = torch.stack((torch.arange(0,num_nodes), torch.arange(0,num_nodes))).type(torch.cuda.LongTensor)
    value_tmp = torch.ones(num_nodes).type(torch.cuda.FloatTensor)
    A.append((edge_tmp, value_tmp))
    
    # 准备数据
    node_features = torch.from_numpy(node_features).type(torch.cuda.FloatTensor)
    train_node = torch.from_numpy(np.array(labels[0])[:,0]).type(torch.cuda.LongTensor)
    train_target = torch.from_numpy(np.array(labels[0])[:,1]).type(torch.cuda.LongTensor)
    valid_node = torch.from_numpy(np.array(labels[1])[:,0]).type(torch.cuda.LongTensor)
    valid_target = torch.from_numpy(np.array(labels[1])[:,1]).type(torch.cuda.LongTensor)
    test_node = torch.from_numpy(np.array(labels[2])[:,0]).type(torch.cuda.LongTensor)
    test_target = torch.from_numpy(np.array(labels[2])[:,1]).type(torch.cuda.LongTensor)
    
    num_classes = np.max([torch.max(train_target).item(), 
                         torch.max(valid_target).item(), 
                         torch.max(test_target).item()]) + 1
    
    # 初始化模型
    model = GTN(num_edge=len(A),
                num_channels=args.num_channels,
                w_in=node_features.shape[1],
                w_out=args.node_dim,
                num_class=num_classes,
                num_nodes=num_nodes,
                num_layers=args.num_layers,
                args=args)
    model.cuda()
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # 训练循环
    best_val_f1 = 0
    best_Ws = None
    
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        
        # 前向传播
        loss, y_train, Ws = model(A, node_features, train_node, train_target)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 验证
        model.eval()
        with torch.no_grad():
            _, y_valid, _ = model(A, node_features, valid_node, valid_target)
            _, y_test, _ = model(A, node_features, test_node, test_target)
            
            # 计算F1分数
            train_f1 = f1_score(train_target.cpu(), y_train.argmax(dim=1).cpu(), average='micro')
            valid_f1 = f1_score(valid_target.cpu(), y_valid.argmax(dim=1).cpu(), average='micro')
            test_f1 = f1_score(test_target.cpu(), y_test.argmax(dim=1).cpu(), average='micro')
            
            print(f'Epoch {epoch:03d}: Train F1: {train_f1:.4f}, Valid F1: {valid_f1:.4f}, Test F1: {test_f1:.4f}')
            
            # 保存最佳模型
            if valid_f1 > best_val_f1:
                best_val_f1 = valid_f1
                best_Ws = Ws
                
                # 保存metapath图结构
                if args.save_metapath:
                    save_metapath_graph(A, best_Ws, num_nodes, 
                                      os.path.join(args.save_dir, args.dataset))
                    print(f'Metapath graph structure saved to {os.path.join(args.save_dir, args.dataset)}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='DBLP',
                        help='Dataset name')
    parser.add_argument('--model', type=str, default='GTN',
                        help='Model name')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs to train')
    parser.add_argument('--node_dim', type=int, default=64,
                        help='Node embedding dimension')
    parser.add_argument('--num_channels', type=int, default=2,
                        help='Number of channels')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.001,
                        help='Weight decay')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='Number of GT layers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--save_metapath', action='store_true',
                        help='Save metapath graph structure')
    parser.add_argument('--save_dir', type=str, default='metapath_data',
                        help='Directory to save metapath graph')
    
    args = parser.parse_args()
    train(args) 