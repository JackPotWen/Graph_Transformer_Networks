import pickle
import numpy as np
import scipy.sparse as sp
import os

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
    # 获取当前文件的绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 获取项目根目录
    root_dir = os.path.abspath(os.path.join(current_dir, "../../../"))
    # 设置数据路径
    data_dir = os.path.join(root_dir, "data", "DBLP")
    edges_file = os.path.join(data_dir, "edges.pkl")
    
    try:
        with open(edges_file, 'rb') as f:
            edges_data = pickle.load(f)
        
        print(f"成功加载edges.pkl文件")
        print(f"文件中包含 {len(edges_data)} 个矩阵")
        
        # 分析每个矩阵
        for i, matrix in enumerate(edges_data):
            matrix_name = f"矩阵 {i+1}"
            analyze_sparse_matrix(matrix, matrix_name)
            
    except Exception as e:
        print(f"处理文件时出错:")
        print(f"错误信息: {str(e)}")

if __name__ == "__main__":
    main() 