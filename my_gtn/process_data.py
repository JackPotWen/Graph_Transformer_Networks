import pickle
import numpy as np
import os
import shutil

def load_metapath_data(data_path):
    """加载元路径数据"""
    with open(os.path.join(data_path, 'metapath_edges.pkl'), 'rb') as f:
        metapath_edges = pickle.load(f)
    with open(os.path.join(data_path, 'metapath_weights.pkl'), 'rb') as f:
        metapath_weights = pickle.load(f)
    return metapath_edges, metapath_weights

def filter_empty_metapaths(metapath_edges, metapath_weights):
    """过滤掉没有边的元路径，保持权重矩阵不变"""
    filtered_edges = []
    empty_count = 0
    
    # 只过滤边矩阵，保持权重矩阵不变
    for edge_matrix in metapath_edges:
        if len(edge_matrix.data) > 0:  # 只保留有边的元路径
            filtered_edges.append(edge_matrix)
        else:
            empty_count += 1
    
    # 权重矩阵保持不变
    return filtered_edges, metapath_weights, empty_count

def save_processed_data(data, save_path, filename):
    """保存处理后的数据"""
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, filename), 'wb') as f:
        pickle.dump(data, f)

def main():
    # 设置路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)
    source_path = os.path.join(root_dir, 'metapath_data', 'DBLP')
    target_path = os.path.join(root_dir, 'metapath_data', 'processed_DBLP')
    
    # 加载原始数据
    print("正在加载原始元路径数据...")
    metapath_edges, metapath_weights = load_metapath_data(source_path)
    
    # 打印原始数据信息
    print(f"\n原始数据统计:")
    print(f"元路径总数: {len(metapath_edges)}")
    print(f"权重矩阵数量: {len(metapath_weights)}")
    print("\n权重矩阵形状:")
    for i, w in enumerate(metapath_weights):
        print(f"权重矩阵 {i+1}: {w.shape}")
    
    # 过滤空元路径
    print("\n正在过滤空元路径...")
    filtered_edges, filtered_weights, empty_count = filter_empty_metapaths(metapath_edges, metapath_weights)
    
    # 打印过滤后的信息
    print(f"\n过滤后数据统计:")
    print(f"删除的空元路径数量: {empty_count}")
    print(f"保留的元路径数量: {len(filtered_edges)}")
    print(f"保留的权重矩阵数量: {len(filtered_weights)}")
    print("\n过滤后权重矩阵形状:")
    for i, w in enumerate(filtered_weights):
        print(f"权重矩阵 {i+1}: {w.shape}")
    
    # 保存处理后的数据
    print("\n正在保存处理后的数据...")
    save_processed_data(filtered_edges, target_path, 'processed_metapath_edges.pkl')
    save_processed_data(filtered_weights, target_path, 'processed_metapath_weights.pkl')
    
    # 复制原始数据文件（除了已处理的文件）
    print("\n正在复制其他数据文件...")
    for file in os.listdir(source_path):
        if file not in ['metapath_edges.pkl', 'metapath_weights.pkl']:
            src_file = os.path.join(source_path, file)
            dst_file = os.path.join(target_path, file)
            if os.path.isfile(src_file):
                shutil.copy2(src_file, dst_file)
    
    print(f"\n处理完成！数据已保存到: {target_path}")
    print("处理后的文件:")
    print(f"- processed_metapath_edges.pkl")
    print(f"- processed_metapath_weights.pkl")

if __name__ == "__main__":
    main() 