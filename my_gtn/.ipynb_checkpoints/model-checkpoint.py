import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.utils import add_self_loops
from torch_scatter import scatter_add
import torch_sparse

class GTConv(nn.Module):
    """图转换层，用于学习metapath结构"""
    def __init__(self, in_channels, out_channels, num_nodes):
        super(GTConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_nodes = num_nodes
        # 修改权重矩阵的维度，确保与输入边的数量匹配
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels))
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        
    def forward(self, A, num_nodes=None):
        if num_nodes is None:
            num_nodes = self.num_nodes
            
        # 对权重进行softmax归一化
        weight = F.softmax(self.weight, dim=1)  # [out_channels, in_channels]
        
        # 构建新的邻接矩阵
        results = []
        for i in range(self.out_channels):
            for j, (edge_index, edge_value) in enumerate(A):
                if j == 0:
                    total_edge_index = edge_index
                    # 确保权重维度正确
                    total_edge_value = edge_value * weight[i, j].item()
                else:
                    total_edge_index = torch.cat((total_edge_index, edge_index), dim=1)
                    total_edge_value = torch.cat((total_edge_value, edge_value * weight[i, j].item()))
            
            # 合并重复边
            index, value = torch_sparse.coalesce(total_edge_index.detach(), 
                                               total_edge_value, 
                                               m=num_nodes, 
                                               n=num_nodes, 
                                               op='add')
            results.append((index, value))
            
        return results, weight

class GTLayer(nn.Module):
    """GT层，包含两个GTConv用于构建metapath"""
    def __init__(self, in_channels, out_channels, num_nodes, first=True):
        super(GTLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.first = first
        self.num_nodes = num_nodes
        
        if self.first:
            self.conv1 = GTConv(in_channels, out_channels, num_nodes)
            self.conv2 = GTConv(in_channels, out_channels, num_nodes)
        else:
            self.conv1 = GTConv(in_channels, out_channels, num_nodes)
    
    def forward(self, A, num_nodes=None, H_=None):
        if num_nodes is None:
            num_nodes = self.num_nodes
            
        if self.first:
            result_A, W1 = self.conv1(A, num_nodes)
            result_B, W2 = self.conv2(A, num_nodes)
            W = [W1, W2]
        else:
            result_A = H_
            result_B, W1 = self.conv1(A, num_nodes)
            W = [W1]
            
        # 构建新的图结构
        H = []
        for i in range(len(result_A)):
            a_edge, a_value = result_A[i]
            b_edge, b_value = result_B[i]
            
            # 构建稀疏矩阵并相乘
            mat_a = torch.sparse_coo_tensor(a_edge, a_value, (num_nodes, num_nodes))
            mat_b = torch.sparse_coo_tensor(b_edge, b_value, (num_nodes, num_nodes))
            mat = torch.sparse.mm(mat_a, mat_b).coalesce()
            
            edges, values = mat.indices(), mat.values()
            H.append((edges, values))
            
        return H, W

class GTN(nn.Module):
    """Graph Transformer Network主模型"""
    def __init__(self, num_edge, num_channels, w_in, w_out, num_class, num_nodes, num_layers, args=None):
        super(GTN, self).__init__()
        self.num_edge = num_edge
        self.num_channels = num_channels
        self.num_nodes = num_nodes
        self.w_in = w_in
        self.w_out = w_out
        self.num_class = num_class
        self.num_layers = num_layers
        self.args = args
        
        # 构建GT层
        self.layers = nn.ModuleList([
            GTLayer(num_edge, num_channels, num_nodes, first=(i==0))
            for i in range(num_layers)
        ])
        
        # 输出层
        self.gcn = nn.Linear(w_in, w_out)
        self.linear = nn.Linear(w_out * num_channels, num_class)
        
        # 损失函数
        if args and args.dataset == 'PPI':
            self.loss_fn = nn.BCELoss()
            self.activation = nn.Sigmoid()
        else:
            self.loss_fn = nn.CrossEntropyLoss()
            self.activation = None
            
    def normalization(self, H, num_nodes):
        """对图结构进行归一化"""
        norm_H = []
        for i in range(self.num_channels):
            edge, value = H[i]
            deg_row, deg_col = self.norm(edge.detach(), num_nodes, value)
            value = deg_row * value
            norm_H.append((edge, value))
        return norm_H
    
    def norm(self, edge_index, num_nodes, edge_weight):
        """计算归一化系数"""
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)
            
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        
        return deg_inv_sqrt[row], deg_inv_sqrt[col]
    
    def forward(self, A, X, target_x, target=None, num_nodes=None, eval=False):
        """前向传播"""
        if num_nodes is None:
            num_nodes = self.num_nodes
            
        # 存储所有层的W矩阵
        all_Ws = []
        
        # 通过GT层
        H = None
        for i in range(self.num_layers):
            if i == 0:
                H, W = self.layers[i](A, num_nodes)
            else:
                H, W = self.layers[i](A, num_nodes, H)
            H = self.normalization(H, num_nodes)
            all_Ws.extend(W)
            
        # 通过GCN层
        X_ = []
        for i in range(self.num_channels):
            edge_index, edge_weight = H[i]
            x = self.gcn(X)
            x = F.relu(x)
            X_.append(x)
            
        # 合并多通道特征
        if self.num_channels > 1:
            X_ = torch.cat(X_, dim=1)
        else:
            X_ = X_[0]
            
        # 输出层
        y = self.linear(X_[target_x])
        
        if eval:
            return y
            
        if target is not None:
            if self.activation is not None:
                y = self.activation(y)
            loss = self.loss_fn(y, target)
            return loss, y, all_Ws
            
        return y, all_Ws 