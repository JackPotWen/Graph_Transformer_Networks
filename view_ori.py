import pickle
import numpy as np
import scipy.sparse as sp
import os

def load_metapath_data(data_path):
    """加载元路径数据"""
    with open(os.path.join(data_path, 'metapath_edges.pkl'), 'rb') as f:
        metapath_edges = pickle.load(f)
    with open(os.path.join(data_path, 'metapath_weights.pkl'), 'rb') as f:
        metapath_weights = pickle.load(f)
    return metapath_edges, metapath_weights

def print_metapath_info(metapath_edges, metapath_weights):
    """打印每个元路径的基本信息"""
    print("="*50)
    print("DBLP元路径特征统计信息 (原始模型)")
    print("="*50)
    
    # 定义元路径类型（根据DBLP数据集的特点）
    metapath_types = [
        "APA",  # Author-Paper-Author
        "APCPA",  # Author-Paper-Conference-Paper-Author
        "APTPA",  # Author-Paper-Term-Paper-Author
        "APVPA"   # Author-Paper-Venue-Paper-Author
    ]
    
    print(f"\n总共生成了 {len(metapath_edges)} 个元路径")
    print("="*50)
    
    # 打印权重矩阵信息
    print("\n权重矩阵信息:")
    print("-"*30)
    for i, W_list in enumerate(metapath_weights):
        print(f"\n第 {i+1} 层权重矩阵:")
        for j, W in enumerate(W_list):
            print(f"\n  权重矩阵 {j+1}:")
            print(f"  形状: {W.shape}")
            print(f"  非零元素数量: {np.count_nonzero(W)}")
            print(f"  稀疏度: {np.count_nonzero(W) / W.size:.6f}")
            if np.count_nonzero(W) > 0:
                print(f"  最大值: {np.max(W):.4f}")
                print(f"  最小值: {np.min(W):.4f}")
                print(f"  均值: {np.mean(W):.4f}")
                print(f"  标准差: {np.std(W):.4f}")
    
    print("\n元路径边信息:")
    print("="*50)
    
    for i, edge_matrix in enumerate(metapath_edges):
        print(f"\n元路径 {i+1}")
        print("-"*30)
        
        # 获取权重统计信息
        weights = edge_matrix.data
        
        # 打印基本信息
        print(f"矩阵形状: {edge_matrix.shape}")
        print(f"非零元素数量: {edge_matrix.nnz}")
        print(f"稀疏度: {edge_matrix.nnz / (edge_matrix.shape[0] * edge_matrix.shape[1]):.6f}")
        
        # 检查是否有边
        if len(weights) > 0:
            max_weight = np.max(weights)
            min_weight = np.min(weights)
            mean_weight = np.mean(weights)
            std_weight = np.std(weights)
            median_weight = np.median(weights)
            
            print(f"\n权重统计:")
            print(f"  最大值: {max_weight:.4f}")
            print(f"  最小值: {min_weight:.4f}")
            print(f"  均值: {mean_weight:.4f}")
            print(f"  标准差: {std_weight:.4f}")
            print(f"  中位数: {median_weight:.4f}")
            
            # 打印一些示例边（前5个非零元素）
            print("\n示例边（前5个）:")
            rows, cols = edge_matrix.nonzero()
            values = edge_matrix.data
            for j in range(min(5, len(rows))):
                print(f"  节点 {rows[j]} -> 节点 {cols[j]}: {values[j]:.4f}")
        else:
            print("\n该元路径没有边（权重数组为空）")

def main():
    # 设置数据路径 - 使用原始模型的路径
    data_path = os.path.join('metapath_data', 'original', 'DBLP')
    
    # 加载数据
    print("正在加载原始模型的元路径数据...")
    print(f"数据路径: {data_path}")
    metapath_edges, metapath_weights = load_metapath_data(data_path)
    
    # 打印信息
    print_metapath_info(metapath_edges, metapath_weights)

if __name__ == "__main__":
    main() 