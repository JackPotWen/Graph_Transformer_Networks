import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

"""
    异构图Transformer层和模型
    基于原Graph Transformer，适配异构图特性，使用PyG
"""

class HeterogeneousMultiHeadAttentionLayer(nn.Module):
    """异构图多头注意力层 - PyG版本"""
    
    def __init__(self, in_dim, out_dim, num_heads, use_bias=False, edge_feat_dim=1):
        super().__init__()
        
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.edge_feat_dim = edge_feat_dim
        
        # 查询、键、值的线性变换
        if use_bias:
            self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.K = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=True)
        else:
            self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.K = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=False)
        
        # 边特征处理（如果有的话）
        if edge_feat_dim > 0:
            self.edge_proj = nn.Linear(edge_feat_dim, num_heads, bias=False)
    
    def forward(self, x, edge_index, edge_attr=None):
        """前向传播"""
        # 计算查询、键、值
        Q = self.Q(x).view(-1, self.num_heads, self.out_dim)
        K = self.K(x).view(-1, self.num_heads, self.out_dim)
        V = self.V(x).view(-1, self.num_heads, self.out_dim)
        
        # 获取源节点和目标节点
        row, col = edge_index
        
        # 计算注意力分数
        score = (K[row] * Q[col]).sum(dim=-1, keepdim=True)  # [num_edges, num_heads, 1]
        
        # 如果有边特征，将其加入到注意力分数中
        if edge_attr is not None and edge_attr.size(-1) == self.edge_feat_dim:
            edge_attention = self.edge_proj(edge_attr)  # [num_edges, num_heads]
            score = score.squeeze(-1) + edge_attention
            score = score.unsqueeze(-1)
        
        # 缩放和指数化
        score = torch.exp((score / np.sqrt(self.out_dim)).clamp(-5, 5))
        
        # 计算注意力权重
        score = score.squeeze(-1)  # [num_edges, num_heads]
        
        # 聚合邻居信息
        out = torch.zeros_like(V)  # [num_nodes, num_heads, out_dim]
        
        # 使用scatter_add进行聚合
        for h in range(self.num_heads):
            weighted_values = V[row, h] * score[:, h].unsqueeze(-1)  # [num_edges, out_dim]
            out[:, h] = torch.zeros_like(V[:, h]).scatter_add_(0, col, weighted_values)
            
            # 归一化
            norm = torch.zeros_like(out[:, h]).scatter_add_(0, col, score[:, h])
            norm = torch.clamp(norm, min=1e-8)
            out[:, h] = out[:, h] / norm.unsqueeze(-1)
        
        return out

class HeterogeneousGraphTransformerLayer(nn.Module):
    """异构图Transformer层 - PyG版本"""
    
    def __init__(self, in_dim, out_dim, num_heads, dropout=0.0, layer_norm=False, 
                 batch_norm=True, residual=True, use_bias=False, edge_feat_dim=1):
        super().__init__()

        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm        
        self.batch_norm = batch_norm
        
        # 多头注意力层
        self.attention = HeterogeneousMultiHeadAttentionLayer(
            in_dim, out_dim//num_heads, num_heads, use_bias, edge_feat_dim
        )
        
        # 输出投影
        self.O = nn.Linear(out_dim, out_dim)

        # 归一化层
        if self.layer_norm:
            self.layer_norm1 = nn.LayerNorm(out_dim)
        if self.batch_norm:
            self.batch_norm1 = nn.BatchNorm1d(out_dim)
        
        # 前馈网络
        self.FFN_layer1 = nn.Linear(out_dim, out_dim*2)
        self.FFN_layer2 = nn.Linear(out_dim*2, out_dim)

        if self.layer_norm:
            self.layer_norm2 = nn.LayerNorm(out_dim)
        if self.batch_norm:
            self.batch_norm2 = nn.BatchNorm1d(out_dim)
        
    def forward(self, x, edge_index, edge_attr=None):
        """前向传播"""
        h_in1 = x # 第一个残差连接
        
        # 多头注意力输出
        attn_out = self.attention(x, edge_index, edge_attr)
        h = attn_out.view(-1, self.out_channels)
        
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.O(h)
        
        if self.residual:
            h = h_in1 + h # 残差连接
        
        if self.layer_norm:
            h = self.layer_norm1(h)
        if self.batch_norm:
            h = self.batch_norm1(h)
        
        h_in2 = h # 第二个残差连接
        
        # 前馈网络
        h = self.FFN_layer1(h)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_layer2(h)

        if self.residual:
            h = h_in2 + h # 残差连接
        
        if self.layer_norm:
            h = self.layer_norm2(h)
        if self.batch_norm:
            h = self.batch_norm2(h)       

        return h
        
    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, heads={}, residual={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels, 
            self.num_heads, self.residual
        )

class MLPReadout(nn.Module):
    """MLP读出层"""
    
    def __init__(self, input_dim, output_dim, L=2): # L=nb_hidden_layers
        super().__init__()
        list_FC_layers = [nn.Linear(input_dim//2**l, input_dim//2**(l+1), bias=True) for l in range(L)]
        list_FC_layers.append(nn.Linear(input_dim//2**L, output_dim, bias=True))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L
        
    def forward(self, x):
        y = x
        for l in range(self.L):
            y = self.FC_layers[l](y)
            y = F.relu(y)
        y = self.FC_layers[self.L](y)
        return y

class HeterogeneousGraphTransformerNet(nn.Module):
    """异构图Transformer网络 - PyG版本"""
    
    def __init__(self, net_params):
        super().__init__()

        in_dim_node = net_params['in_dim'] # 节点特征维度
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        n_classes = net_params['n_classes']
        num_heads = net_params['n_heads']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        edge_feat_dim = net_params.get('edge_feat_dim', 1)

        self.readout = net_params['readout']
        self.layer_norm = net_params['layer_norm']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.dropout = dropout
        self.n_classes = n_classes
        self.device = net_params['device']
        self.lap_pos_enc = net_params.get('lap_pos_enc', False)
        self.wl_pos_enc = net_params.get('wl_pos_enc', False)
        max_wl_role_index = 100 
        
        # 位置编码
        if self.lap_pos_enc:
            pos_enc_dim = net_params['pos_enc_dim']
            self.embedding_lap_pos_enc = nn.Linear(pos_enc_dim, hidden_dim)
        if self.wl_pos_enc:
            self.embedding_wl_pos_enc = nn.Embedding(max_wl_role_index, hidden_dim)
        
        # 节点特征嵌入
        if net_params.get('node_feat_is_int', False):
            # 如果节点特征是整数（如类别特征）
            self.embedding_h = nn.Embedding(in_dim_node, hidden_dim)
        else:
            # 如果节点特征是连续值
            self.embedding_h = nn.Linear(in_dim_node, hidden_dim)
        
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        
        # 异构图Transformer层
        self.layers = nn.ModuleList([
            HeterogeneousGraphTransformerLayer(
                hidden_dim, hidden_dim, num_heads, dropout, 
                self.layer_norm, self.batch_norm, self.residual,
                edge_feat_dim=edge_feat_dim
            ) for _ in range(n_layers-1)
        ])
        
        # 最后一层
        self.layers.append(
            HeterogeneousGraphTransformerLayer(
                hidden_dim, out_dim, num_heads, dropout, 
                self.layer_norm, self.batch_norm, self.residual,
                edge_feat_dim=edge_feat_dim
            )
        )
        
        # 输出层
        self.MLP_layer = MLPReadout(out_dim, n_classes)

    def forward(self, data, h_lap_pos_enc=None, h_wl_pos_enc=None):
        """前向传播"""
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # 输入嵌入
        if hasattr(self.embedding_h, 'weight'):  # Embedding层
            h = self.embedding_h(x)
        else:  # Linear层
            h = self.embedding_h(x)
            
        # 位置编码
        if self.lap_pos_enc and h_lap_pos_enc is not None:
            h_lap_pos_enc = self.embedding_lap_pos_enc(h_lap_pos_enc.float()) 
            h = h + h_lap_pos_enc
        if self.wl_pos_enc and h_wl_pos_enc is not None:
            h_wl_pos_enc = self.embedding_wl_pos_enc(h_wl_pos_enc) 
            h = h + h_wl_pos_enc
            
        h = self.in_feat_dropout(h)
        
        # 异构图Transformer层
        for conv in self.layers:
            h = conv(h, edge_index, edge_attr)
            
        # 输出
        h_out = self.MLP_layer(h)

        return h_out
    
    def loss(self, pred, label):
        """损失函数"""
        # 计算标签权重用于加权损失计算
        V = label.size(0)
        label_count = torch.bincount(label)
        label_count = label_count[label_count.nonzero()].squeeze()
        cluster_sizes = torch.zeros(self.n_classes).long().to(self.device)
        cluster_sizes[torch.unique(label)] = label_count
        weight = (V - cluster_sizes).float() / V
        weight *= (cluster_sizes>0).float()
        
        # 加权交叉熵用于不平衡类别
        criterion = nn.CrossEntropyLoss(weight=weight)
        loss = criterion(pred, label)

        return loss

def accuracy(pred, target):
    """计算准确率"""
    return (pred.argmax(dim=1) == target).float().mean().item() 