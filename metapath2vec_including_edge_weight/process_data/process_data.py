import pickle
import numpy as np
import scipy.sparse as sp
import os
import argparse
from tqdm import tqdm

# 节点类型范围定义
NODE_RANGES = {
    'A': (0, 4056),      # Author: 0-4056
    'P': (4057, 18384),  # Paper: 4057-18384
    'C': (18385, 18404)  # Conference: 18385-18404
}

# 边类型定义
EDGE_TYPES = ['PA', 'AP', 'PC', 'CP', 'AA', 'PP', 'CC', 'AC', 'CA']

def get_node_type(node_id):
    """根据节点ID判断节点类型"""
    for node_type, (start, end) in NODE_RANGES.items():
        if start <= node_id <= end:
            return node_type
    raise ValueError(f"Invalid node ID: {node_id}")

def get_edge_type(src_id, dst_id):
    """根据源节点和目标节点ID判断边类型"""
    src_type = get_node_type(src_id)
    dst_type = get_node_type(dst_id)
    return f"{src_type}{dst_type}"

# def process_edges(edges_data, num_nodes=18405):
#     """处理边数据，生成9种类型的稀疏矩阵"""
#     # 初始化9个稀疏矩阵（CSR格式）
#     matrices = [sp.csr_matrix((num_nodes, num_nodes), dtype=np.float32) for _ in range(len(EDGE_TYPES))]
    
#     # 处理两个channel的数据
#     for channel_idx, (edge_indices, edge_weights) in enumerate(edges_data):
#         print(f"\n处理 Channel {channel_idx + 1}/2:")
#         # 转置边索引矩阵以便于处理
#         edge_indices = edge_indices.T  # shape: (num_edges, 2)
        
#         # 使用tqdm创建进度条
#         for i in tqdm(range(edge_indices.shape[0]), desc="处理边", unit="边"):
#             src_id, dst_id = edge_indices[i]
#             weight = edge_weights[i]
            
#             # 获取边类型
#             edge_type = get_edge_type(src_id, dst_id)
#             if edge_type in EDGE_TYPES:
#                 # 找到对应的矩阵索引
#                 matrix_idx = EDGE_TYPES.index(edge_type)
#                 # 更新矩阵（如果是第二个channel，取平均值）
#                 if channel_idx == 0:
#                     matrices[matrix_idx][src_id, dst_id] = weight
#                 else:
#                     matrices[matrix_idx][src_id, dst_id] = (matrices[matrix_idx][src_id, dst_id] + weight) / 2
    
#     return matrices

def process_edges_fast(edges_data, num_nodes=18405):
    """更快地处理边数据，生成9种类型的稀疏矩阵"""
    edge_data_dict = {etype: [] for etype in EDGE_TYPES}

    for channel_idx, (edge_indices, edge_weights) in enumerate(edges_data):
        print(f"\n处理 Channel {channel_idx + 1}/2:")
        edge_indices = edge_indices.T

        for i in tqdm(range(edge_indices.shape[0]), desc="收集边", unit="边"):
            src_id, dst_id = edge_indices[i]
            weight = edge_weights[i]
            edge_type = get_edge_type(src_id, dst_id)
            if edge_type in EDGE_TYPES:
                edge_data_dict[edge_type].append((src_id, dst_id, weight, channel_idx))

    print("\n开始构建稀疏矩阵...")
    matrices = []
    for etype in EDGE_TYPES:
        srcs_ch1, dsts_ch1, vals_ch1 = [], [], []
        srcs_ch2, dsts_ch2, vals_ch2 = [], [], []

        for src, dst, val, ch in edge_data_dict[etype]:
            if ch == 0:
                srcs_ch1.append(src)
                dsts_ch1.append(dst)
                vals_ch1.append(val)
            else:
                srcs_ch2.append(src)
                dsts_ch2.append(dst)
                vals_ch2.append(val)

        # 构建两个 channel 的矩阵
        coo1 = sp.coo_matrix((vals_ch1, (srcs_ch1, dsts_ch1)), shape=(num_nodes, num_nodes))
        coo2 = sp.coo_matrix((vals_ch2, (srcs_ch2, dsts_ch2)), shape=(num_nodes, num_nodes))

        # 若两个都有，则做平均；否则保留现有
        if coo1.nnz > 0 and coo2.nnz > 0:
            mat = (coo1.tocsr() + coo2.tocsr()) / 2
        elif coo1.nnz > 0:
            mat = coo1.tocsr()
        else:
            mat = coo2.tocsr()

        matrices.append(mat)

    return matrices
def main():
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='处理DBLP边数据')
    parser.add_argument('--epoch', type=int, required=True, help='要处理的epoch编号')
    args = parser.parse_args()
    
    # 获取当前文件的绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 获取项目根目录（向上导航三层：process_data -> metapath2vec_including_edge_weight -> Graph_Transformer_Networks）
    root_dir = os.path.abspath(os.path.join(current_dir, "../../"))
    
    # 设置输入输出路径
    input_dir = os.path.join(root_dir, "metapath_data", "H", "DBLP", "H_Epoch")
    output_dir = os.path.join(current_dir, "save_data")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    input_file = os.path.join(input_dir, f"metapath_H_Epoch{args.epoch}.pkl")
    output_file = os.path.join(output_dir, f"Epoch{args.epoch}_edges.pkl")
    
    print(f"正在处理 Epoch {args.epoch} 的数据")
    print(f"项目根目录: {root_dir}")
    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    
    try:
        # 读取数据
        print("\n正在加载数据...")
        with open(input_file, 'rb') as f:
            edges_data = pickle.load(f)
        
        # 处理数据
        print("\n开始处理数据...")
        processed_matrices = process_edges_fast(edges_data)
        
        # 保存处理后的数据
        print("\n正在保存数据...")
        with open(output_file, 'wb') as f:
            pickle.dump(processed_matrices, f)
        
        # 打印处理结果统计
        print("\n处理完成！结果统计：")
        for i, matrix in enumerate(processed_matrices):
            print(f"\n{EDGE_TYPES[i]} 边:")
            print(f"非零元素数量: {matrix.nnz}")
            print(f"稀疏度: {1 - matrix.nnz / (matrix.shape[0] * matrix.shape[1]):.4%}")
            if matrix.nnz > 0:
                print(f"权重范围: [{matrix.data.min():.6f}, {matrix.data.max():.6f}]")
                print(f"平均权重: {matrix.data.mean():.6f}")
        
    except Exception as e:
        print(f"\n处理文件时出错:")
        print(f"错误信息: {str(e)}")

if __name__ == "__main__":
    main() 