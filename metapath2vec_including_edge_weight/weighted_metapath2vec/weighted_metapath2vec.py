from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torch.nn import Embedding
from torch.utils.data import DataLoader

from torch_geometric.index import index2ptr
from torch_geometric.typing import EdgeType, NodeType, OptTensor
from torch_geometric.utils import sort_edge_index

EPS = 1e-15

class WeightedMetaPath2Vec(torch.nn.Module):
    """带权重的MetaPath2Vec模型实现
    
    参数:
        edge_index_dict (Dict[Tuple[str, str, str], torch.Tensor]): 存储异构图每种边类型的边索引的字典
        edge_weight_dict (Dict[Tuple[str, str, str], torch.Tensor]): 存储异构图每种边类型的边权重的字典
        embedding_dim (int): 嵌入向量的维度
        metapath (List[Tuple[str, str, str]]): 元路径，表示为 (源节点类型, 关系类型, 目标节点类型) 元组的列表
        walk_length (int): 随机游走的长度
        context_size (int): 正样本的上下文窗口大小
        walks_per_node (int, optional): 每个节点的采样游走次数 (默认: 1)
        num_negative_samples (int, optional): 每个正样本的负样本数量 (默认: 1)
        num_nodes_dict (Dict[str, int], optional): 存储每种节点类型节点数量的字典 (默认: None)
        sparse (bool, optional): 如果设置为 True，权重矩阵的梯度将是稀疏的 (默认: False)
        neg_sample_lambda (float, optional): 负采样中随机采样的比例，范围[0,1] (默认: 0.3)
    """
    def __init__(
        self,
        edge_index_dict: Dict[EdgeType, Tensor],
        edge_weight_dict: Dict[EdgeType, Tensor],
        embedding_dim: int,
        metapath: List[EdgeType],
        walk_length: int,
        context_size: int,
        walks_per_node: int = 1,
        num_negative_samples: int = 1,
        num_nodes_dict: Optional[Dict[NodeType, int]] = None,
        sparse: bool = False,
        neg_sample_lambda: float = 0.5,
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

        # 预处理边索引和权重
        self.rowptr_dict, self.col_dict, self.rowcount_dict = {}, {}, {}
        self.weight_dict = {}
        for keys, edge_index in edge_index_dict.items():
            sizes = (num_nodes_dict[keys[0]], num_nodes_dict[keys[-1]])
            row, col = sort_edge_index(edge_index, num_nodes=max(sizes)).cpu()
            rowptr = index2ptr(row, size=sizes[0])
            self.rowptr_dict[keys] = rowptr
            self.col_dict[keys] = col
            self.rowcount_dict[keys] = rowptr[1:] - rowptr[:-1]
            
            # 处理权重
            weights = edge_weight_dict[keys]
            if weights.dim() == 1:
                weights = weights[:rowptr.size(0)-1]  # 只取到rowptr的倒数第二个元素
            self.weight_dict[keys] = weights

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

        # 验证lambda参数
        if not 0 <= neg_sample_lambda <= 1:
            raise ValueError("neg_sample_lambda必须在[0,1]范围内")
        self.neg_sample_lambda = neg_sample_lambda

        self.reset_parameters()

    def reset_parameters(self):
        """重置所有可学习参数"""
        self.embedding.reset_parameters()

    def forward(self, node_type: str, batch: OptTensor = None) -> Tensor:
        """返回指定节点类型的节点嵌入"""
        emb = self.embedding.weight[self.start[node_type]:self.end[node_type]]
        return emb if batch is None else emb.index_select(0, batch)

    def loader(self, **kwargs):
        """返回数据加载器，用于创建正负随机游走"""
        return DataLoader(range(self.num_nodes_dict[self.metapath[0][0]]),
                         collate_fn=self._sample, **kwargs)

    def _pos_sample(self, batch: Tensor) -> Tensor:
        """生成正样本随机游走，考虑边权重"""
        batch = batch.repeat(self.walks_per_node)

        rws = [batch]
        for i in range(self.walk_length):
            edge_type = self.metapath[i % len(self.metapath)]
            batch = weighted_sample(
                self.rowptr_dict[edge_type],
                self.col_dict[edge_type],
                self.rowcount_dict[edge_type],
                self.weight_dict[edge_type],
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
        """生成负样本随机游走，使用混合采样策略
        
        混合采样策略：P(v) = λ·random + (1-λ)·inv_weight
        其中：
        - λ (neg_sample_lambda) 控制随机采样的比例
        - inv_weight 是基于边权重的反向采样概率
        
        参数:
            batch (Tensor): 当前批次的节点索引
            
        返回:
            Tensor: 负样本随机游走序列
        """
        batch = batch.repeat(self.walks_per_node * self.num_negative_samples)

        rws = [batch]
        for i in range(self.walk_length):
            edge_type = self.metapath[i % len(self.metapath)]
            num_nodes = self.num_nodes_dict[edge_type[-1]]
            
            # 获取当前节点的所有可能邻居及其权重
            start = self.rowptr_dict[edge_type][batch.clamp(0, len(self.rowptr_dict[edge_type])-2)]
            end = self.rowptr_dict[edge_type][(batch + 1).clamp(0, len(self.rowptr_dict[edge_type])-1)]
            
            # 为每个节点生成负样本
            neg_samples = []
            for j, (s, e) in enumerate(zip(start, end)):
                if s < e and s < len(self.col_dict[edge_type]):  # 确保索引有效
                    # 获取当前节点的所有邻居及其权重
                    neighbors = self.col_dict[edge_type][s:e]
                    weights = self.weight_dict[edge_type][s:e]
                    
                    if len(neighbors) > 0 and len(weights) > 0:  # 确保有邻居和权重
                        try:
                            # 决定是否使用随机采样
                            if torch.rand(1, device=batch.device) < self.neg_sample_lambda:
                                # 随机采样
                                idx = torch.randint(0, len(neighbors), (1,), device=batch.device)
                            else:
                                # 基于权重的反向采样
                                inv_weights = 1.0 / (weights + EPS)  # 避免除零
                                inv_weights = inv_weights / inv_weights.sum()  # 归一化
                                
                                if torch.isfinite(inv_weights).all():  # 确保权重有效
                                    idx = torch.multinomial(inv_weights, 1, replacement=True)
                                else:
                                    # 如果权重无效，使用均匀分布
                                    idx = torch.randint(0, len(neighbors), (1,), device=batch.device)
                            
                            neg_samples.append(neighbors[idx])
                        except RuntimeError:
                            # 如果采样失败，使用均匀分布
                            idx = torch.randint(0, len(neighbors), (1,), device=batch.device)
                            neg_samples.append(neighbors[idx])
                    else:
                        # 如果没有邻居或权重，随机采样一个有效节点
                        neg_samples.append(torch.randint(0, num_nodes, (1,), device=batch.device))
                else:
                    # 如果索引无效，随机采样一个有效节点
                    neg_samples.append(torch.randint(0, num_nodes, (1,), device=batch.device))
            
            batch = torch.cat(neg_samples, dim=0)
            # 确保batch中的索引不会越界
            batch = batch.clamp(0, num_nodes - 1)
            rws.append(batch)

        rw = torch.stack(rws, dim=-1)
        rw.add_(self.offset.view(1, -1))

        walks = []
        num_walks_per_rw = 1 + self.walk_length + 1 - self.context_size
        for j in range(num_walks_per_rw):
            walks.append(rw[:, j:j + self.context_size])
        return torch.cat(walks, dim=0)

    def _sample(self, batch: List[int]) -> Tuple[Tensor, Tensor]:
        """采样正负样本"""
        if not isinstance(batch, Tensor):
            batch = torch.tensor(batch, dtype=torch.long)
        return self._pos_sample(batch), self._neg_sample(batch)

    def loss(self, pos_rw: Tensor, neg_rw: Tensor) -> Tensor:
        """计算损失函数"""
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
        """通过逻辑回归下游任务评估嵌入质量"""
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


def weighted_sample(rowptr: Tensor, col: Tensor, rowcount: Tensor, weights: Tensor,
                   subset: Tensor, num_neighbors: int, dummy_idx: int) -> Tensor:
    """考虑边权重的邻居节点采样
    
    参数:
        rowptr (Tensor): 行指针
        col (Tensor): 列索引
        rowcount (Tensor): 每行的邻居数量
        weights (Tensor): 边权重
        subset (Tensor): 要采样的节点子集
        num_neighbors (int): 每个节点要采样的邻居数量
        dummy_idx (int): 虚拟节点索引
        
    返回:
        Tensor: 采样得到的邻居节点
    """
    mask = subset >= dummy_idx
    subset = subset.clamp(min=0, max=rowptr.numel() - 2)
    
    # 获取每个节点的邻居权重
    start = rowptr[subset]
    end = rowptr[subset + 1]
    count = end - start
    
    # 计算每个节点的权重分布
    probs = []
    for i in range(len(subset)):
        if count[i] > 0:
            node_weights = weights[start[i]:end[i]]
            probs.append(node_weights / node_weights.sum())
        else:
            probs.append(torch.tensor([1.0], device=subset.device))
    
    # 根据权重分布进行采样
    sampled_neighbors = []
    for i, prob in enumerate(probs):
        if count[i] > 0:
            idx = torch.multinomial(prob, num_neighbors, replacement=True)
            sampled_neighbors.append(col[start[i] + idx])
        else:
            sampled_neighbors.append(torch.full((num_neighbors,), dummy_idx, 
                                              device=subset.device))
    
    sampled = torch.stack(sampled_neighbors)
    sampled[mask] = dummy_idx
    return sampled 