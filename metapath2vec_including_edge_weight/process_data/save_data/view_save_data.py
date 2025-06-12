import pickle
import numpy as np
import scipy.sparse as sp
import os
import argparse

def main():
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='查看处理后的DBLP边数据')
    parser.add_argument('--epoch', type=int, required=True, help='要查看的epoch编号')
    args = parser.parse_args()
    
    # 获取当前文件的绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 获取数据文件路径
    data_file = os.path.join(current_dir, f"Epoch{args.epoch}_edges.pkl")
    
    print(f"数据文件: {data_file}")
    print("="*50)
    
    try:
        # 读取数据
        with open(data_file, 'rb') as f:
            matrices = pickle.load(f)
        
        # 边类型定义
        edge_types = ['PA', 'AP', 'PC', 'CP', 'AA', 'PP', 'CC', 'AC', 'CA']
        
        # 打印总体信息
        print(f"\n类型: {type(matrices)}")
        print(f"值: {matrices}")
        
        # 打印每个矩阵的基本信息
        print("\n各矩阵信息:")
        for i, matrix in enumerate(matrices):
            print(f"\n{edge_types[i]} 边矩阵:")
            print(f"类型: {type(matrix)}")
            print(f"形状: {matrix.shape}")
            print(f"非零元素数: {matrix.nnz}")
            print(f"存储格式: {'CSR' if isinstance(matrix, sp.csr_matrix) else 'CSC'}")
            
    except Exception as e:
        print(f"\n处理文件时出错:")
        print(f"错误信息: {str(e)}")

if __name__ == "__main__":
    main() 