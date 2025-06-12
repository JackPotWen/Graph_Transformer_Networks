import torch
import torch.nn as nn
import torch.nn.functional as F
from gcn import GCNConv

class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=1, dropout=0.5, args=None):
        super(GCN, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.args = args
        
        # 创建多层GCN
        self.convs = nn.ModuleList()
        if num_layers == 1:
            self.convs.append(GCNConv(in_channels, out_channels, args=args))
        else:
            self.convs.append(GCNConv(in_channels, hidden_channels, args=args))
            for _ in range(num_layers - 2):
                self.convs.append(GCNConv(hidden_channels, hidden_channels, args=args))
            self.convs.append(GCNConv(hidden_channels, out_channels, args=args))
        
        # 损失函数
        if args.dataset in ["PPI", "BOOK", "MUSIC"]:
            self.m = nn.Sigmoid()
            self.loss = nn.BCELoss()
        else:
            self.loss = nn.CrossEntropyLoss()

    def forward(self, x, edge_index, edge_weight, target_x, target, eval=False):
        # 多层GCN前向传播
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_weight)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 获取目标节点的预测结果
        y = x[target_x]
        
        if eval:
            return y
        else:
            if self.args.dataset in ["PPI", "BOOK", "MUSIC"]:
                loss = self.loss(self.m(y), target)
            else:
                loss = self.loss(y, target)
            return loss, y, None  # 返回None作为Ws以保持与GTN接口一致 