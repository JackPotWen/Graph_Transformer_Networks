import torch
import numpy as np
import torch.nn as nn
import pickle
import argparse
from sklearn.metrics import f1_score as sk_f1_score
from utils import init_seed, _norm
import copy

#数据集查看
with open('data/ACM/node_features.pkl', 'rb') as f:
    node_features = pickle.load(f)
with open('data/ACM/edges.pkl', 'rb') as f:
    edges = pickle.load(f)
with open('data/ACM/labels.pkl', 'rb') as f:
    labels = pickle.load(f)

num_nodes = edges[0].shape[0]
print(labels)
#print(num_nodes)
#print(edges,'\n')
"""
[
 <CSR matrix>, <CSC matrix>,
 <CSR matrix>, <CSC matrix>
]
"""
#print(node_features)
#print(labels)
#print(f"边数量: {len(edges)}")
#print(f"节点特征形状: {node_features.shape}")

#制作4种边的邻接矩阵并在最后加上self-loop
"""
A = []

for i,edge in enumerate(edges):
    edge_tmp = torch.from_numpy(np.vstack((edge.nonzero()[1], edge.nonzero()[0]))).type(torch.cuda.LongTensor)
    value_tmp = torch.ones(edge_tmp.shape[1]).type(torch.cuda.FloatTensor)
    A.append((edge_tmp,value_tmp))
edge_tmp = torch.stack((torch.arange(0, num_nodes), torch.arange(0, num_nodes))).type(torch.cuda.LongTensor)
value_tmp = torch.ones(num_nodes).type(torch.cuda.FloatTensor)
A.append((edge_tmp, value_tmp))

num_edge_type = len(A)
print(num_edge_type)
print(A[0])
"""