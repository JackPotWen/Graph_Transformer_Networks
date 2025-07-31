import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, scatter

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
        
        # 获取当前子图的节点数量
        num_nodes = x.size(0)
        
        # 用list收集每个head的结果，最后stack，避免原地操作
        out_list = []
        for h in range(self.num_heads):
            weighted_values = V[row, h] * score[:, h].unsqueeze(-1)  # [num_edges, out_dim]
            out_h = scatter(weighted_values, col, dim=0, dim_size=num_nodes, reduce='sum')
            norm = scatter(score[:, h], col, dim=0, dim_size=num_nodes, reduce='sum')
            norm = torch.clamp(norm, min=1e-8)
            out_h = out_h / norm.unsqueeze(-1)
            out_list.append(out_h)
        out = torch.stack(out_list, dim=1)  # [num_nodes, num_heads, out_dim]
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
        
        # 如果输入输出维度不同，添加投影层用于残差连接
        if in_dim != out_dim and residual:
            self.residual_proj = nn.Linear(in_dim, out_dim)
        else:
            self.residual_proj = None
        
    def forward(self, x, edge_index, edge_attr=None):
        """前向传播"""
        h_in1 = x # 第一个残差连接
        
        # 多头注意力输出
        attn_out = self.attention(x, edge_index, edge_attr)
        h = attn_out.view(-1, self.out_channels)
        
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.O(h)
        
        if self.residual:
            if self.residual_proj is not None:
                h = self.residual_proj(h_in1) + h # 残差连接（带投影）
            else:
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
        self.metapath2vec_pos_enc = net_params.get('metapath2vec_pos_enc', False)
        # PE融合模式控制 - 在这里直接设置，不需要修改配置文件
        self.pe_fusion_mode = net_params.get('pe_fusion_mode', 'add')  # 'add' 或 'concat'
        max_wl_role_index = 100 
        
        # 位置编码
        if self.lap_pos_enc:
            pos_enc_dim = net_params['pos_enc_dim']
            self.embedding_lap_pos_enc = nn.Linear(pos_enc_dim, hidden_dim)
        if self.wl_pos_enc:
            self.embedding_wl_pos_enc = nn.Embedding(max_wl_role_index, hidden_dim)
        if self.metapath2vec_pos_enc:
            # metapath2vec PE的维度需要从数据中获取，这里先设置为一个默认值
            # 实际维度会在forward中根据输入数据调整
            self.embedding_metapath2vec_pos_enc = None  # 将在forward中动态创建
        
        # 为concat模式准备投影层 - 预先注册为nn.Module子模块
        self.concat_projection = None  # 将在forward中动态创建并注册
        
        # 节点特征嵌入
        if net_params.get('node_feat_is_int', False):
            # 如果节点特征是整数（如类别特征）
            self.embedding_h = nn.Embedding(in_dim_node, hidden_dim)
        else:
            # 如果节点特征是连续值
            self.embedding_h = nn.Linear(in_dim_node, hidden_dim)
        
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        
        # 异构图Transformer层 - 确保所有层使用相同的隐藏维度
        self.layers = nn.ModuleList([
            HeterogeneousGraphTransformerLayer(
                hidden_dim, hidden_dim, num_heads, dropout, 
                self.layer_norm, self.batch_norm, self.residual,
                edge_feat_dim=edge_feat_dim
            ) for _ in range(n_layers-1)
        ])
        
        # 最后一层 - 输出维度为out_dim
        self.layers.append(
            HeterogeneousGraphTransformerLayer(
                hidden_dim, out_dim, num_heads, dropout, 
                self.layer_norm, self.batch_norm, self.residual,
                edge_feat_dim=edge_feat_dim
            )
        )
        
        # 输出层
        self.MLP_layer = MLPReadout(out_dim, n_classes)

    def forward(self, data, h_lap_pos_enc=None, h_wl_pos_enc=None, h_metapath2vec_pos_enc=None):
        """前向传播"""
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # 输入嵌入
        if hasattr(self.embedding_h, 'weight'):  # Embedding层
            h = self.embedding_h(x)
        else:  # Linear层
            h = self.embedding_h(x)
            
        # 位置编码处理
        pe_features = []
        
        if self.lap_pos_enc and h_lap_pos_enc is not None:
            h_lap_pos_enc = self.embedding_lap_pos_enc(h_lap_pos_enc.float())
            pe_features.append(h_lap_pos_enc)
            
        if self.wl_pos_enc and h_wl_pos_enc is not None:
            h_wl_pos_enc = self.embedding_wl_pos_enc(h_wl_pos_enc)
            pe_features.append(h_wl_pos_enc)
            
        if self.metapath2vec_pos_enc and h_metapath2vec_pos_enc is not None:
            # 动态创建metapath2vec PE的嵌入层（如果还没有创建）
            if self.embedding_metapath2vec_pos_enc is None:
                pe_dim = h_metapath2vec_pos_enc.size(-1)
                self.embedding_metapath2vec_pos_enc = nn.Linear(pe_dim, self.layers[0].in_channels).to(h.device)
            h_metapath2vec_pos_enc = self.embedding_metapath2vec_pos_enc(h_metapath2vec_pos_enc.float())
            pe_features.append(h_metapath2vec_pos_enc)
        
        # PE融合模式控制 - 在这里直接设置，不需要修改配置文件
        # 根据融合模式处理位置编码
        if pe_features:
            if self.pe_fusion_mode == 'add':
                # 加法模式：将所有PE相加后加到节点特征上
                # 公式: h = h_node + h_pe
                combined_pe = sum(pe_features)
                h = h + combined_pe
            elif self.pe_fusion_mode == 'concat':
                # 拼接模式：将节点特征和PE拼接
                # 公式: h = [h_node, h_pe]
                combined_pe = torch.cat(pe_features, dim=-1)
                h = torch.cat([h, combined_pe], dim=-1)
                # 动态创建并注册投影层为nn.Module子模块
                original_in_dim = self.layers[0].in_channels
                new_in_dim = h.size(-1)
                if new_in_dim != original_in_dim:
                    if self.concat_projection is None:
                        # 创建投影层并注册为模型的一部分
                        self.concat_projection = nn.Linear(new_in_dim, original_in_dim).to(h.device)
                        # 将投影层注册为nn.Module的子模块，这样它会被包含在模型参数中
                        self.add_module('concat_projection', self.concat_projection)
                    h = self.concat_projection(h)
            
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