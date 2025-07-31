import pickle
import numpy as np
import torch
import scipy.sparse as sp
from scipy.sparse import csr_matrix, csc_matrix
import os
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix
from scipy.sparse.linalg import eigsh
import torch.nn.functional as F

def load_metapath2vec_pe(pe_path):
    """
    加载预生成的metapath2vec位置编码
    
    Args:
        pe_path: metapath2vec嵌入文件路径
        
    Returns:
        metapath2vec_pe: metapath2vec位置编码张量
    """
    print(f"正在加载metapath2vec位置编码: {pe_path}")
    
    try:
        with open(pe_path, 'rb') as f:
            vec_feature = pickle.load(f)
        
        # 转换为torch张量
        if isinstance(vec_feature, np.ndarray):
            metapath2vec_pe = torch.FloatTensor(vec_feature)
        else:
            metapath2vec_pe = torch.FloatTensor(vec_feature)
        
        print(f"metapath2vec位置编码加载完成: 形状 {metapath2vec_pe.shape}")
        return metapath2vec_pe
        
    except Exception as e:
        print(f"加载metapath2vec位置编码失败: {e}")
        return None

def load_dblp_data(data_path="/home/kuei-jan/github/Graph_Transformer_Networks/data/DBLP"):
    """
    加载DBLP异构图数据
    
    Args:
        data_path: 数据文件路径
        
    Returns:
        node_features: 节点特征矩阵
        edges_list: 边矩阵列表
        labels: 标签字典，包含train_mask, val_mask, test_mask
    """
    print("正在加载DBLP异构图数据...")
    
    # 加载节点特征
    with open(os.path.join(data_path, "node_features.pkl"), 'rb') as f:
        node_features = pickle.load(f)
    print(f"节点特征形状: {node_features.shape}")
    
    # 加载边数据
    with open(os.path.join(data_path, "edges.pkl"), 'rb') as f:
        edges_list = pickle.load(f)
    print(f"边矩阵数量: {len(edges_list)}")
    
    # 加载标签（包含训练、验证、测试分割）
    with open(os.path.join(data_path, "labels.pkl"), 'rb') as f:
        labels = pickle.load(f)
    print(f"标签类型: {type(labels)}")
    if isinstance(labels, list):
        print(f"标签长度: {len(labels)}")
    else:
        print(f"标签内容: {labels}")
    
    return node_features, edges_list, labels

def merge_heterogeneous_edges(edges_list, remove_duplicates=True):
    """
    将多个异构图边矩阵进行拼接和去重处理
    
    Args:
        edges_list: 边矩阵列表，每个矩阵代表一种边类型
        remove_duplicates: 是否去除重复边
        
    Returns:
        merged_adj: 合并后的邻接矩阵
        edge_types: 边类型信息字典
    """
    print("正在合并异构图边矩阵...")
    
    if not edges_list:
        raise ValueError("边矩阵列表为空")
    
    # 获取图的节点数量
    n_nodes = edges_list[0].shape[0]
    print(f"节点数量: {n_nodes}")
    
    # 初始化合并后的邻接矩阵
    merged_adj = sp.csr_matrix((n_nodes, n_nodes), dtype=np.float32)
    
    # 记录每种边类型的统计信息
    edge_types = {}
    
    for i, edge_matrix in enumerate(edges_list):
        print(f"处理边矩阵 {i+1}: 形状 {edge_matrix.shape}, 非零元素 {edge_matrix.nnz}")
        
        # 确保矩阵是CSR格式
        if not isinstance(edge_matrix, csr_matrix):
            edge_matrix = edge_matrix.tocsr()
        
        # 将当前边矩阵添加到合并矩阵中
        merged_adj += edge_matrix.astype(np.float32)
        
        # 记录边类型信息
        edge_types[f'type_{i}'] = {
            'matrix': edge_matrix,
            'nnz': edge_matrix.nnz,
            'density': edge_matrix.nnz / (edge_matrix.shape[0] * edge_matrix.shape[1])
        }
    
    print(f"合并后邻接矩阵非零元素: {merged_adj.nnz}")
    
    if remove_duplicates:
        # 去除重复边（将大于1的值设为1）
        merged_adj.data = np.minimum(merged_adj.data, 1.0)
        merged_adj.eliminate_zeros()
        print(f"去重后邻接矩阵非零元素: {merged_adj.nnz}")
    
    return merged_adj, edge_types

def parse_label_list(labels, num_nodes):
    """
    labels: list of 3 lists, each (N, 2) or (N,2) array, [node_id, label]
    返回: y, train_mask, val_mask, test_mask
    """
    y = torch.full((num_nodes,), -1, dtype=torch.long)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    for idx, (split, mask) in enumerate(zip(labels, [train_mask, val_mask, test_mask])):
        arr = np.array(split)
        node_ids = arr[:,0].astype(int)
        node_labels = arr[:,1].astype(int)
        mask[node_ids] = True
        y[node_ids] = torch.from_numpy(node_labels)
    return y, train_mask, val_mask, test_mask

def create_pyg_graph(adj_matrix, node_features, labels):
    """
    创建PyG图对象
    
    Args:
        adj_matrix: 邻接矩阵
        node_features: 节点特征
        labels: 标签字典，包含train_mask, val_mask, test_mask
        
    Returns:
        data: PyG Data对象
    """
    print("正在创建PyG图对象...")
    
    # 转换为PyG格式
    edge_index, edge_weight = from_scipy_sparse_matrix(adj_matrix)
    
    # 转换节点特征
    if isinstance(node_features, np.ndarray):
        node_features = torch.FloatTensor(node_features)
    
    # 创建PyG Data对象
    num_nodes = node_features.shape[0]
    # 适配labels为list的情况
    if isinstance(labels, list) and len(labels) == 3:
        y, train_mask, val_mask, test_mask = parse_label_list(labels, num_nodes)
    elif isinstance(labels, dict):
        y = torch.LongTensor(labels['labels'] if 'labels' in labels else labels['y'])
        train_mask = torch.BoolTensor(labels['train_mask'])
        val_mask = torch.BoolTensor(labels['val_mask'])
        test_mask = torch.BoolTensor(labels['test_mask'])
    else:
        y = torch.LongTensor(labels)
        train_mask = val_mask = test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    data = Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_weight.unsqueeze(-1) if edge_weight is not None else torch.ones(edge_index.size(1), 1),
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask
    )
    
    print(f"PyG图创建完成: 节点数 {data.x.size(0)}, 边数 {data.edge_index.size(1)}")
    print(f"训练节点: {data.train_mask.sum()}, 验证节点: {data.val_mask.sum()}, 测试节点: {data.test_mask.sum()}")
    
    return data

def prepare_heterogeneous_data(data_path="/home/kuei-jan/github/Graph_Transformer_Networks/data/DBLP", 
                             add_lap_pos_enc=False, pos_enc_dim=8,
                             add_metapath2vec_pe=False, metapath2vec_pe_paths=None, metapath_names=None):
    """
    准备异构图数据的完整流程
    
    Args:
        data_path: 数据文件路径
        add_lap_pos_enc: 是否添加拉普拉斯位置编码
        pos_enc_dim: 位置编码维度
        add_metapath2vec_pe: 是否添加metapath2vec位置编码
        metapath2vec_pe_paths: metapath2vec嵌入文件路径列表
        metapath_names: 对应的元路径名称列表
        
    Returns:
        data: PyG Data对象
        edge_types: 边类型信息
    """
    # 加载原始数据
    node_features, edges_list, labels = load_dblp_data(data_path)
    
    # 合并边矩阵
    merged_adj, edge_types = merge_heterogeneous_edges(edges_list)
    
    # 创建PyG图
    data = create_pyg_graph(merged_adj, node_features, labels)
    
    # 如果启用拉普拉斯位置编码，则添加位置编码
    if add_lap_pos_enc:
        data = add_laplacian_positional_encoding(data, pos_enc_dim)
    
    # 如果启用metapath2vec位置编码，则添加位置编码
    if add_metapath2vec_pe and metapath2vec_pe_paths:
        data = add_multiple_metapath2vec_positional_encoding(data, metapath2vec_pe_paths, metapath_names)
    
    return data, edge_types

def compute_laplacian_positional_encoding(adj_matrix, pos_enc_dim=8):
    """
    计算拉普拉斯位置编码
    
    Args:
        adj_matrix: 邻接矩阵 (scipy.sparse格式)
        pos_enc_dim: 位置编码维度
        
    Returns:
        lap_pos_enc: 拉普拉斯位置编码矩阵 (n_nodes, pos_enc_dim)
    """
    print(f"正在计算拉普拉斯位置编码 (维度: {pos_enc_dim})...")
    
    # 确保邻接矩阵是CSR格式
    if not isinstance(adj_matrix, csr_matrix):
        adj_matrix = adj_matrix.tocsr()
    
    n_nodes = adj_matrix.shape[0]
    
    # 计算度矩阵
    deg = np.array(adj_matrix.sum(axis=1)).flatten()
    
    # 计算拉普拉斯矩阵 L = D - A
    # 使用对称归一化拉普拉斯矩阵 L_sym = I - D^(-1/2) * A * D^(-1/2)
    deg_inv_sqrt = np.power(deg, -0.5)
    deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0
    
    # 构建对称归一化拉普拉斯矩阵
    D_inv_sqrt = sp.diags(deg_inv_sqrt)
    L_sym = sp.eye(n_nodes) - D_inv_sqrt @ adj_matrix @ D_inv_sqrt
    
    # 计算特征值和特征向量
    # 使用eigsh计算最小的k个特征值（不包括0特征值）
    try:
        eigenvalues, eigenvectors = eigsh(L_sym, k=pos_enc_dim+1, sigma=0, which='LM')
        # 去掉第一个特征向量（对应特征值0）
        lap_pos_enc = eigenvectors[:, 1:pos_enc_dim+1]
    except:
        # 如果eigsh失败，使用完整的特征值分解
        print("警告: eigsh失败，使用完整特征值分解...")
        eigenvalues, eigenvectors = np.linalg.eigh(L_sym.toarray())
        # 选择最小的非零特征值对应的特征向量
        non_zero_indices = np.where(eigenvalues > 1e-8)[0]
        if len(non_zero_indices) >= pos_enc_dim:
            lap_pos_enc = eigenvectors[:, non_zero_indices[:pos_enc_dim]]
        else:
            # 如果非零特征值不够，用零填充
            lap_pos_enc = np.zeros((n_nodes, pos_enc_dim))
            lap_pos_enc[:, :len(non_zero_indices)] = eigenvectors[:, non_zero_indices]
    
    # 转换为torch张量
    lap_pos_enc = torch.FloatTensor(lap_pos_enc)
    
    print(f"拉普拉斯位置编码计算完成: 形状 {lap_pos_enc.shape}")
    return lap_pos_enc

def add_laplacian_positional_encoding(data, pos_enc_dim=8):
    """
    为PyG Data对象添加拉普拉斯位置编码
    
    Args:
        data: PyG Data对象
        pos_enc_dim: 位置编码维度
        
    Returns:
        data: 添加了位置编码的PyG Data对象
    """
    print("正在添加拉普拉斯位置编码...")
    
    # 从边索引构建邻接矩阵
    edge_index = data.edge_index
    edge_weight = data.edge_attr.squeeze() if data.edge_attr is not None else None
    
    n_nodes = data.x.size(0)
    
    # 构建邻接矩阵
    if edge_weight is not None:
        adj_matrix = sp.coo_matrix((edge_weight.numpy(), 
                                   (edge_index[0].numpy(), edge_index[1].numpy())), 
                                  shape=(n_nodes, n_nodes))
    else:
        adj_matrix = sp.coo_matrix((np.ones(edge_index.size(1)), 
                                   (edge_index[0].numpy(), edge_index[1].numpy())), 
                                  shape=(n_nodes, n_nodes))
    
    # 转换为CSR格式
    adj_matrix = adj_matrix.tocsr()
    
    # 计算拉普拉斯位置编码
    lap_pos_enc = compute_laplacian_positional_encoding(adj_matrix, pos_enc_dim)
    
    # 将位置编码添加到Data对象
    data.lap_pos_enc = lap_pos_enc
    
    print(f"拉普拉斯位置编码已添加到Data对象")
    return data

def add_metapath2vec_positional_encoding(data, metapath2vec_pe_path):
    """
    为PyG Data对象添加metapath2vec位置编码
    
    Args:
        data: PyG Data对象
        metapath2vec_pe_path: metapath2vec嵌入文件路径
        
    Returns:
        data: 添加了metapath2vec位置编码的PyG Data对象
    """
    print("正在添加metapath2vec位置编码...")
    
    # 加载metapath2vec位置编码
    metapath2vec_pe = load_metapath2vec_pe(metapath2vec_pe_path)
    
    if metapath2vec_pe is None:
        print("警告: 无法加载metapath2vec位置编码，跳过添加")
        return data
    
    # 检查维度匹配
    n_nodes = data.x.size(0)
    if metapath2vec_pe.size(0) != n_nodes:
        print(f"警告: metapath2vec位置编码节点数 ({metapath2vec_pe.size(0)}) 与图节点数 ({n_nodes}) 不匹配")
        # 如果节点数不匹配，尝试截取或填充
        if metapath2vec_pe.size(0) > n_nodes:
            metapath2vec_pe = metapath2vec_pe[:n_nodes]
        else:
            # 用零填充
            padding = torch.zeros(n_nodes - metapath2vec_pe.size(0), metapath2vec_pe.size(1))
            metapath2vec_pe = torch.cat([metapath2vec_pe, padding], dim=0)
    
    # 将位置编码添加到Data对象
    data.metapath2vec_pos_enc = metapath2vec_pe
    
    print(f"metapath2vec位置编码已添加到Data对象: 形状 {metapath2vec_pe.shape}")
    return data

def add_multiple_metapath2vec_positional_encoding(data, metapath2vec_pe_paths, metapath_names=None):
    """
    为PyG Data对象添加多个metapath2vec位置编码
    
    Args:
        data: PyG Data对象
        metapath2vec_pe_paths: metapath2vec嵌入文件路径列表
        metapath_names: 对应的元路径名称列表
        
    Returns:
        data: 添加了多个metapath2vec位置编码的PyG Data对象
    """
    print("正在添加多个metapath2vec位置编码...")
    
    if not metapath2vec_pe_paths:
        print("警告: 没有提供metapath2vec PE路径，跳过添加")
        return data
    
    if metapath_names is None:
        metapath_names = [f"metapath_{i}" for i in range(len(metapath2vec_pe_paths))]
    
    # 存储所有加载的PE
    all_pe = []
    loaded_metapaths = []
    
    for i, (pe_path, metapath_name) in enumerate(zip(metapath2vec_pe_paths, metapath_names)):
        print(f"正在加载元路径 {metapath_name} 的PE: {pe_path}")
        
        # 加载metapath2vec位置编码
        metapath2vec_pe = load_metapath2vec_pe(pe_path)
        
        if metapath2vec_pe is None:
            print(f"警告: 无法加载元路径 {metapath_name} 的位置编码，跳过")
            continue
        
        # 检查维度匹配
        n_nodes = data.x.size(0)
        if metapath2vec_pe.size(0) != n_nodes:
            print(f"警告: 元路径 {metapath_name} 位置编码节点数 ({metapath2vec_pe.size(0)}) 与图节点数 ({n_nodes}) 不匹配")
            # 如果节点数不匹配，尝试截取或填充
            if metapath2vec_pe.size(0) > n_nodes:
                metapath2vec_pe = metapath2vec_pe[:n_nodes]
                print(f"  截取PE到 {n_nodes} 个节点")
            else:
                # 用零填充
                padding = torch.zeros(n_nodes - metapath2vec_pe.size(0), metapath2vec_pe.size(1))
                metapath2vec_pe = torch.cat([metapath2vec_pe, padding], dim=0)
                print(f"  填充PE到 {n_nodes} 个节点")
        else:
            print(f"  节点数匹配: {n_nodes}")
        
        all_pe.append(metapath2vec_pe)
        loaded_metapaths.append(metapath_name)
        print(f"元路径 {metapath_name} 位置编码加载完成: 形状 {metapath2vec_pe.shape}")
    
    if not all_pe:
        print("警告: 没有成功加载任何metapath2vec位置编码")
        return data
    
    # 保持PE的独立性，不进行融合
    # 将多个PE存储为列表，让模型处理融合
    data.metapath2vec_pos_enc_list = all_pe  # 存储多个独立的PE
    data.metapath_names = loaded_metapaths  # 记录使用的元路径名称
    data.metapath2vec_pe_paths = metapath2vec_pe_paths  # 记录使用的PE路径
    
    print(f"多个metapath2vec位置编码已添加到Data对象: {len(all_pe)}个独立PE")
    print(f"使用的元路径: {loaded_metapaths}")
    for i, pe in enumerate(all_pe):
        print(f"  PE {i+1} 形状: {pe.shape}")
    return data

if __name__ == "__main__":
    # 测试数据加载
    data, edge_types = prepare_heterogeneous_data()
    print("数据准备完成！")
    print(f"图信息: {data}") 