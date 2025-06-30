from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torch.nn import Embedding
from torch.utils.data import DataLoader, Dataset

from torch_geometric.typing import EdgeType, NodeType, OptTensor
from torch_geometric.utils import sort_edge_index

EPS = 1e-15

class NodeDataset(Dataset):
    """节点数据集，用于DataLoader"""
    def __init__(self, num_nodes: int):
        self.num_nodes = num_nodes
    
    def __len__(self):
        return self.num_nodes
    
    def __getitem__(self, idx):
        return idx

class MetaPath2Vec(torch.nn.Module):
    """MetaPath2Vec 模型实现，基于论文 "metapath2vec: Scalable Representation Learning for Heterogeneous Networks"
    
    参数:
        edge_index_dict (Dict[Tuple[str, str, str], torch.Tensor]): 存储异构图每种边类型的边索引的字典
        embedding_dim (int): 嵌入向量的维度
        metapath (List[Tuple[str, str, str]]): 元路径，表示为 (源节点类型, 关系类型, 目标节点类型) 元组的列表
        walk_length (int): 随机游走的长度
        context_size (int): 正样本的上下文窗口大小
        walks_per_node (int, optional): 每个节点的采样游走次数 (默认: 1)
        num_negative_samples (int, optional): 每个正样本的负样本数量 (默认: 1)
        num_nodes_dict (Dict[str, int], optional): 存储每种节点类型节点数量的字典 (默认: None)
        sparse (bool, optional): 如果设置为 True，权重矩阵的梯度将是稀疏的 (默认: False)
    """
    def __init__(
        self,
        edge_index_dict: Dict[EdgeType, Tensor],
        embedding_dim: int,
        metapath: List[EdgeType],
        walk_length: int,
        context_size: int,
        walks_per_node: int = 1,
        num_negative_samples: int = 1,
        num_nodes_dict: Optional[Dict[NodeType, int]] = None,
        sparse: bool = False,
    ):
        super().__init__()

        # 如果没有提供节点数量字典，则从边索引中推断
        if num_nodes_dict is None:
            num_nodes_dict = {}
            for keys, edge_index in edge_index_dict.items():
                key = keys[0]
                N = int(edge_index[0].max() + 1)
                num_nodes_dict[key] = max(N, num_nodes_dict.get(key, N))

                key = keys[-1]
                N = int(edge_index[1].max() + 1)
                num_nodes_dict[key] = max(N, num_nodes_dict.get(key, N))

        # 预处理边索引，使用邻接表而不是指针
        self.adj_dict = {}  # 邻接表字典
        for keys, edge_index in edge_index_dict.items():
            sizes = (num_nodes_dict[keys[0]], num_nodes_dict[keys[-1]])
            row, col = sort_edge_index(edge_index, num_nodes=max(sizes)).cpu()
            
            # 构建邻接表
            adj_list = [[] for _ in range(sizes[0])]
            
            # 填充邻接表
            for i in range(row.size(0)):
                src_node = int(row[i])
                dst_node = int(col[i])
                
                if src_node < sizes[0]:  # 确保索引在有效范围内
                    adj_list[src_node].append(dst_node)
            
            # 转换为张量
            max_neighbors = max(len(neighbors) for neighbors in adj_list) if adj_list else 0
            if max_neighbors == 0:
                max_neighbors = 1
            
            # 填充邻接表，使用-1表示无效邻居
            adj_tensor = torch.full((sizes[0], max_neighbors), -1, dtype=torch.long)
            
            for i, neighbors in enumerate(adj_list):
                if neighbors:
                    adj_tensor[i, :len(neighbors)] = torch.tensor(neighbors, dtype=torch.long)
            
            self.adj_dict[keys] = adj_tensor

        # 验证元路径的有效性
        for edge_type1, edge_type2 in zip(metapath[:-1], metapath[1:]):
            if edge_type1[-1] != edge_type2[0]:
                raise ValueError(
                    "发现无效的元路径。确保所有连续边类型的目标节点类型与源节点类型匹配。")

        assert walk_length + 1 >= context_size
        if walk_length > len(metapath) and metapath[0][0] != metapath[-1][-1]:
            raise AttributeError(
                "'walk_length' 大于给定的 'metapath'，但 'metapath' 不构成循环")

        self.embedding_dim = embedding_dim
        self.metapath = metapath
        self.walk_length = walk_length
        self.context_size = context_size
        self.walks_per_node = walks_per_node
        self.num_negative_samples = num_negative_samples
        self.num_nodes_dict = num_nodes_dict

        # 计算节点类型的起始和结束索引
        types = {x[0] for x in metapath} | {x[-1] for x in metapath}
        types = sorted(list(types))

        count = 0
        self.start, self.end = {}, {}
        for key in types:
            self.start[key] = count
            count += num_nodes_dict[key]
            self.end[key] = count

        # 计算偏移量
        offset = [self.start[metapath[0][0]]]
        offset += [self.start[keys[-1]] for keys in metapath] * int((walk_length / len(metapath)) + 1)
        offset = offset[:walk_length + 1]
        assert len(offset) == walk_length + 1
        self.offset = torch.tensor(offset)

        # 创建嵌入层，+1 表示用于孤立节点的虚拟节点
        self.embedding = Embedding(count + 1, embedding_dim, sparse=sparse)
        self.dummy_idx = count

        self.reset_parameters()

    def reset_parameters(self):
        """重置所有可学习参数"""
        self.embedding.reset_parameters()

    def forward(self, node_type: str, batch: OptTensor = None) -> Tensor:
        """返回指定节点类型的节点嵌入
        
        参数:
            node_type (str): 节点类型
            batch (OptTensor, optional): 批处理索引
            
        返回:
            Tensor: 节点嵌入
        """
        emb = self.embedding.weight[self.start[node_type]:self.end[node_type]]
        return emb if batch is None else emb.index_select(0, batch)

    def loader(self, **kwargs):
        """返回数据加载器，用于创建正负随机游走
        
        参数:
            **kwargs: DataLoader 的参数，如 batch_size, shuffle 等
        """
        dataset = NodeDataset(self.num_nodes_dict[self.metapath[0][0]])
        return DataLoader(dataset, collate_fn=self._sample, **kwargs)

    def _pos_sample(self, batch: Tensor) -> Tensor:
        """生成正样本随机游走"""
        batch = batch.repeat(self.walks_per_node)

        rws = [batch]
        for i in range(self.walk_length):
            edge_type = self.metapath[i % len(self.metapath)]
            batch = sample_adj(
                self.adj_dict[edge_type],
                batch,
                num_neighbors=1,
                dummy_idx=self.dummy_idx,
            ).view(-1)
            rws.append(batch)

        rw = torch.stack(rws, dim=-1)
        rw.add_(self.offset.view(1, -1))
        rw[rw > self.dummy_idx] = self.dummy_idx

        walks = []
        num_walks_per_rw = 1 + self.walk_length + 1 - self.context_size
        for j in range(num_walks_per_rw):
            walks.append(rw[:, j:j + self.context_size])
        return torch.cat(walks, dim=0)

    def _neg_sample(self, batch: Tensor) -> Tensor:
        """生成负样本随机游走"""
        batch = batch.repeat(self.walks_per_node * self.num_negative_samples)

        rws = [batch]
        for i in range(self.walk_length):
            keys = self.metapath[i % len(self.metapath)]
            batch = torch.randint(0, self.num_nodes_dict[keys[-1]],
                                (batch.size(0), ), dtype=torch.long)
            rws.append(batch)

        rw = torch.stack(rws, dim=-1)
        rw.add_(self.offset.view(1, -1))

        walks = []
        num_walks_per_rw = 1 + self.walk_length + 1 - self.context_size
        for j in range(num_walks_per_rw):
            walks.append(rw[:, j:j + self.context_size])
        return torch.cat(walks, dim=0)

    def _sample(self, batch) -> Tuple[Tensor, Tensor]:
        """采样正负样本"""
        if isinstance(batch, (list, tuple)):
            batch = batch[0]  # Dataset返回的是单个值
        if not isinstance(batch, Tensor):
            batch = torch.tensor(batch, dtype=torch.long)
        return self._pos_sample(batch), self._neg_sample(batch)

    def loss(self, pos_rw: Tensor, neg_rw: Tensor) -> Tensor:
        """计算损失函数
        
        参数:
            pos_rw (Tensor): 正样本随机游走
            neg_rw (Tensor): 负样本随机游走
            
        返回:
            Tensor: 损失值
        """
        # 正样本损失
        start, rest = pos_rw[:, 0], pos_rw[:, 1:].contiguous()

        h_start = self.embedding(start).view(pos_rw.size(0), 1, self.embedding_dim)
        h_rest = self.embedding(rest.view(-1)).view(pos_rw.size(0), -1, self.embedding_dim)

        out = (h_start * h_rest).sum(dim=-1).view(-1)
        pos_loss = -torch.log(torch.sigmoid(out) + EPS).mean()

        # 负样本损失
        start, rest = neg_rw[:, 0], neg_rw[:, 1:].contiguous()

        h_start = self.embedding(start).view(neg_rw.size(0), 1, self.embedding_dim)
        h_rest = self.embedding(rest.view(-1)).view(neg_rw.size(0), -1, self.embedding_dim)

        out = (h_start * h_rest).sum(dim=-1).view(-1)
        neg_loss = -torch.log(1 - torch.sigmoid(out) + EPS).mean()

        return pos_loss + neg_loss

    def test(self, train_z: Tensor, train_y: Tensor, test_z: Tensor,
             test_y: Tensor, solver: str = "lbfgs", *args, **kwargs) -> float:
        """通过逻辑回归下游任务评估嵌入质量
        
        参数:
            train_z (Tensor): 训练集嵌入
            train_y (Tensor): 训练集标签
            test_z (Tensor): 测试集嵌入
            test_y (Tensor): 测试集标签
            solver (str): 优化器类型
            *args, **kwargs: 传递给 LogisticRegression 的参数
            
        返回:
            float: 测试集准确率
        """
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(*args, solver=solver,
                               **kwargs).fit(train_z.detach().cpu().numpy(),
                                           train_y.detach().cpu().numpy())
        return clf.score(test_z.detach().cpu().numpy(),
                        test_y.detach().cpu().numpy())

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'{self.embedding.weight.size(0) - 1}, '
                f'{self.embedding.weight.size(1)})')


def sample_adj(adj: Tensor, subset: Tensor, num_neighbors: int, dummy_idx: int) -> Tensor:
    """邻居节点采样（基于邻接表）
    
    参数:
        adj (Tensor): 邻接表，形状为 (num_nodes, max_neighbors)
        subset (Tensor): 要采样的节点子集
        num_neighbors (int): 每个节点要采样的邻居数量
        dummy_idx (int): 虚拟节点索引
        
    返回:
        Tensor: 采样得到的邻居节点
    """
    # 处理无效节点
    mask = subset >= dummy_idx
    subset = subset.clamp(min=0, max=adj.size(0) - 1)
    
    # 获取每个节点的邻居
    node_neighbors = adj[subset]  # (batch_size, max_neighbors)
    
    # 创建有效邻居的掩码（-1表示无效邻居）
    valid_mask = node_neighbors >= 0
    
    # 采样邻居
    sampled_neighbors = []
    for i in range(len(subset)):
        if mask[i]:
            # 无效节点，返回虚拟节点
            sampled_neighbors.append(torch.full((num_neighbors,), dummy_idx, 
                                              device=subset.device))
        else:
            # 获取当前节点的有效邻居
            valid_neighbors = node_neighbors[i][valid_mask[i]]
            
            if len(valid_neighbors) > 0:
                # 随机采样邻居
                idx = torch.randint(0, len(valid_neighbors), (num_neighbors,))
                sampled_neighbors.append(valid_neighbors[idx])
            else:
                # 没有邻居，返回虚拟节点
                sampled_neighbors.append(torch.full((num_neighbors,), dummy_idx, 
                                                  device=subset.device))
    
    sampled = torch.stack(sampled_neighbors)
    return sampled


# 保留原始函数作为备用
def sample(rowptr: Tensor, col: Tensor, rowcount: Tensor, subset: Tensor,
           num_neighbors: int, dummy_idx: int) -> Tensor:
    """采样邻居节点 - 原始版本（基于指针）
    
    参数:
        rowptr (Tensor): 行指针
        col (Tensor): 列索引
        rowcount (Tensor): 每行的邻居数量
        subset (Tensor): 要采样的节点子集
        num_neighbors (int): 每个节点要采样的邻居数量
        dummy_idx (int): 虚拟节点索引
        
    返回:
        Tensor: 采样得到的邻居节点
    """
    mask = subset >= dummy_idx
    subset = subset.clamp(min=0, max=rowptr.numel() - 2)
    count = rowcount[subset]

    rand = torch.rand((subset.size(0), num_neighbors), device=subset.device)
    rand *= count.to(rand.dtype).view(-1, 1)
    rand = rand.to(torch.long) + rowptr[subset].view(-1, 1)
    rand = rand.clamp(max=col.numel() - 1)  # 如果最后一个节点是孤立的

    col = col[rand] if col.numel() > 0 else rand
    col[mask | (count == 0)] = dummy_idx
    return col 