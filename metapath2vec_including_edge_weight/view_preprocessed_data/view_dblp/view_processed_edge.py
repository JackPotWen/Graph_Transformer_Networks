import pickle
import numpy as np
import scipy.sparse as sp
import os
import argparse
import torch

def analyze_matrix(matrix, matrix_name):
    """分析矩阵的详细信息"""
    print(f"\n{'='*50}")
    print(f"矩阵名称: {matrix_name}")
    print(f"{'='*50}")
    
    if isinstance(matrix, tuple):
        print("\n元组结构:")
        for i, item in enumerate(matrix):
            print(f"\n元素 {i+1}:")
            if isinstance(item, (np.ndarray, torch.Tensor)):
                print(f"类型: {type(item)}")
                print(f"形状: {item.shape}")
                print(f"数据类型: {item.dtype}")
                print(f"数值范围: [{item.min()}, {item.max()}]")
                if len(item.shape) == 1:
                    print(f"前5个元素: {item[:5]}")
            elif isinstance(item, sp.spmatrix):
                analyze_sparse_matrix(item, f"稀疏矩阵 {i+1}")
            else:
                print(f"类型: {type(item)}")
                print(f"值: {item}")
    elif isinstance(matrix, sp.spmatrix):
        analyze_sparse_matrix(matrix, matrix_name)
    else:
        print(f"\n类型: {type(matrix)}")
        print(f"值: {matrix}")

def analyze_sparse_matrix(matrix, matrix_name):
    """分析稀疏矩阵的详细信息"""
    print(f"\n{'='*50}")
    print(f"矩阵名称: {matrix_name}")
    print(f"{'='*50}")
    
    # 基本信息
    print(f"\n1. 基本信息:")
    print(f"矩阵类型: {type(matrix)}")
    print(f"矩阵形状: {matrix.shape}")
    print(f"数据类型: {matrix.dtype}")
    print(f"存储格式: {'CSR' if isinstance(matrix, sp.csr_matrix) else 'CSC'}")
    
    # 稀疏性信息
    total_elements = matrix.shape[0] * matrix.shape[1]
    non_zero = matrix.nnz
    sparsity = 1 - (non_zero / total_elements)
    print(f"\n2. 稀疏性信息:")
    print(f"非零元素数量: {non_zero}")
    print(f"总元素数量: {total_elements}")
    print(f"稀疏度: {sparsity:.4%}")
    
    # 统计信息
    if non_zero > 0:
        values = matrix.data
        print(f"\n3. 数值统计:")
        print(f"最小值: {values.min()}")
        print(f"最大值: {values.max()}")
        print(f"平均值: {values.mean():.4f}")
        print(f"标准差: {values.std():.4f}")
        
        # 计算每行/列的非零元素数量
        if isinstance(matrix, sp.csr_matrix):
            row_nnz = np.diff(matrix.indptr)
            print(f"\n4. 行统计:")
            print(f"每行平均非零元素数: {row_nnz.mean():.2f}")
            print(f"最大行非零元素数: {row_nnz.max()}")
            print(f"最小行非零元素数: {row_nnz.min()}")
        else:  # CSC matrix
            col_nnz = np.diff(matrix.indptr)
            print(f"\n4. 列统计:")
            print(f"每列平均非零元素数: {col_nnz.mean():.2f}")
            print(f"最大列非零元素数: {col_nnz.max()}")
            print(f"最小列非零元素数: {col_nnz.min()}")
    
    # 示例数据
    print(f"\n5. 数据示例:")
    if non_zero > 0:
        # 获取前5个非零元素的位置和值
        if isinstance(matrix, sp.csr_matrix):
            for i in range(min(5, non_zero)):
                row = matrix.indices[i]
                col = matrix.indptr[i]
                value = matrix.data[i]
                print(f"位置 ({row}, {col}): {value}")
        else:  # CSC matrix
            for i in range(min(5, non_zero)):
                col = matrix.indices[i]
                row = matrix.indptr[i]
                value = matrix.data[i]
                print(f"位置 ({row}, {col}): {value}")

def main():
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='查看处理后的DBLP边数据信息')
    parser.add_argument('--epoch', type=int, required=True, help='要查看的epoch编号')
    args = parser.parse_args()
    
    # 获取当前文件的绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 获取项目根目录
    root_dir = os.path.abspath(os.path.join(current_dir, "../../../"))
    # 设置数据路径
    data_dir = os.path.join(root_dir, "metapath_data", "H", "DBLP", "H_Epoch")
    edges_file = os.path.join(data_dir, f"metapath_H_Epoch{args.epoch}.pkl")
    
    print(f"正在查看 Epoch {args.epoch} 的边数据")
    print(f"数据文件: {edges_file}")
    
    try:
        with open(edges_file, 'rb') as f:
            edges_data = pickle.load(f)
        
        print(f"成功加载边数据文件")
        if isinstance(edges_data, list):
            print(f"文件中包含 {len(edges_data)} 个矩阵")
            for i, matrix in enumerate(edges_data):
                analyze_matrix(matrix, f"矩阵 {i+1}")
        else:
            analyze_matrix(edges_data, "矩阵")
            
    except Exception as e:
        print(f"处理文件时出错:")
        print(f"错误信息: {str(e)}")

if __name__ == "__main__":
    main() 