import pickle
import argparse
import os
import numpy as np
import torch
import scipy.sparse as sp

def load_H(epoch_num, dataset='DBLP', base_dir='metapath_data/H'):
    """加载指定epoch的H数据"""
    # 使用os.path.normpath确保路径格式正确
    file_path = os.path.normpath(os.path.join(base_dir, dataset, 'H_Epoch', f'metapath_H_Epoch{epoch_num}.pkl'))
    if not os.path.exists(file_path):
        # 尝试使用当前目录作为基准
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, f'metapath_H_Epoch{epoch_num}.pkl')
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"找不到epoch {epoch_num}的数据文件，已尝试以下路径：\n1. {os.path.join(base_dir, dataset, 'H_Epoch', f'metapath_H_Epoch{epoch_num}.pkl')}\n2. {file_path}")
    
    with open(file_path, 'rb') as f:
        H = pickle.load(f)
    return H

def analyze_H(H):
    """分析H数据的统计信息"""
    print("\n=== H Data Analysis ===")
    print(f"Number of channels: {len(H)}")
    
    for i, (edge_index, edge_value) in enumerate(H):
        print(f"\nChannel {i+1}:")
        print(f"Number of edges: {edge_index.shape[1]}")
        print(f"Edge value statistics:")
        print(f"  - Mean: {np.mean(edge_value):.4f}")
        print(f"  - Std: {np.std(edge_value):.4f}")
        print(f"  - Min: {np.min(edge_value):.4f}")
        print(f"  - Max: {np.max(edge_value):.4f}")
        
        # 转换为稀疏矩阵以获取更多信息
        adj = sp.coo_matrix((edge_value, (edge_index[0], edge_index[1])))
        print(f"Matrix shape: {adj.shape}")
        print(f"Matrix density: {(adj.nnz / (adj.shape[0] * adj.shape[1]))*100:.4f}%")
        
        # 计算每个节点的度
        degrees = np.array(adj.sum(axis=1)).flatten()
        print(f"Node degree statistics:")
        print(f"  - Mean degree: {np.mean(degrees):.4f}")
        print(f"  - Max degree: {np.max(degrees)}")
        print(f"  - Min degree: {np.min(degrees)}")
        
        # 显示一些示例边
        print("\nSample edges (first 5):")
        for j in range(min(5, edge_index.shape[1])):
            print(f"  Edge {j+1}: {edge_index[0][j]} -> {edge_index[1][j]} (value: {edge_value[j]:.4f})")

def main():
    parser = argparse.ArgumentParser(description='View H data for a specific epoch')
    parser.add_argument('--epoch', type=int, required=True,
                      help='Epoch number to view')
    parser.add_argument('--dataset', type=str, default='DBLP',
                      help='Dataset name')
    parser.add_argument('--base_dir', type=str, default='metapath_data/H',
                      help='Base directory for data')
    
    args = parser.parse_args()
    
    try:
        # 加载数据
        print(f"Loading H data for epoch {args.epoch}...")
        H = load_H(args.epoch, args.dataset, args.base_dir)
        
        # 分析数据
        analyze_H(H)
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == '__main__':
    main() 

#python view_H.py --epoch 10