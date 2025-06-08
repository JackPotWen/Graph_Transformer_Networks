import os
import pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict

def load_H_data(epoch, dataset='DBLP'):
    """加载指定epoch的H数据"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, 'H_Epoch', f'metapath_H_Epoch{epoch}.pkl')
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到H数据文件: {file_path}")
        
    with open(file_path, 'rb') as f:
        H_data = pickle.load(f)
    return H_data

def load_original_data(dataset='DBLP'):
    """加载原始图数据"""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    data_dir = os.path.join(base_dir, 'data', dataset)
    
    # 加载边数据
    with open(os.path.join(data_dir, 'edges.pkl'), 'rb') as f:
        edges = pickle.load(f)
    
    # 构建邻接矩阵列表
    A = []
    for edge in edges:
        edge_tmp = np.vstack((edge.nonzero()[1], edge.nonzero()[0]))
        A.append(edge_tmp)
    
    return A

def get_H_neighbors(H_data, node_k):
    """获取节点k在H数据中的邻居"""
    # 合并两个channel的边权重
    edge_weights = defaultdict(float)
    edge_count = defaultdict(int)
    
    for channel_H in H_data:
        edge_index, edge_value = channel_H
        for i, (src, dst) in enumerate(zip(edge_index[0], edge_index[1])):
            if src == node_k:  # 只考虑以node_k为起点的边
                edge = (src, dst)
                edge_weights[edge] += edge_value[i]
                edge_count[edge] += 1
    
    # 计算平均权重
    for edge in edge_weights:
        edge_weights[edge] /= edge_count[edge]
    
    return edge_weights

def get_original_neighbors(A, node_k, max_hops=2):
    """获取节点k在原始数据中的二跳邻居"""
    G = nx.DiGraph()
    visited = set([node_k])
    current_level = set([node_k])
    
    # 添加中心节点
    G.add_node(node_k)
    
    # 进行二跳遍历
    for hop in range(max_hops):
        next_level = set()
        for node in current_level:
            # 遍历所有边类型
            for edge_type in A:
                # 获取当前节点的邻居
                neighbors = edge_type[1][edge_type[0] == node]
                for neighbor in neighbors:
                    if neighbor not in visited:
                        G.add_edge(node, neighbor)
                        next_level.add(neighbor)
                        visited.add(neighbor)
        current_level = next_level
    
    return G

def plot_H_graph(edge_weights, node_k, save_path):
    """绘制H数据中的邻居图"""
    G = nx.DiGraph()
    
    # 添加边和权重
    for (src, dst), weight in edge_weights.items():
        G.add_edge(src, dst, weight=weight)
    
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G)
    
    # 绘制节点
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                          node_size=500, alpha=0.6)
    
    # 绘制边
    nx.draw_networkx_edges(G, pos, edge_color='gray', 
                          arrows=True, arrowsize=20)
    
    # 添加节点标签
    nx.draw_networkx_labels(G, pos, font_size=12)
    
    # 添加边权重标签
    edge_labels = {(u, v): f'{d["weight"]:.3f}' 
                  for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, 
                                font_size=8)
    
    plt.title(f'Neighbors of Node {node_k} in H Data')
    plt.axis('off')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_original_graph(G, node_k, save_path):
    """绘制原始数据中的邻居图"""
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G)
    
    # 绘制节点
    nx.draw_networkx_nodes(G, pos, node_color='lightgreen', 
                          node_size=500, alpha=0.6)
    
    # 绘制边
    nx.draw_networkx_edges(G, pos, edge_color='gray', 
                          arrows=True, arrowsize=20)
    
    # 添加节点标签
    nx.draw_networkx_labels(G, pos, font_size=12)
    
    plt.title(f'Two-hop Neighbors of Node {node_k} in Original Graph')
    plt.axis('off')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def analyze_node(epoch, node_k):
    """分析指定节点在H数据和原始数据中的邻居关系"""
    # 创建保存目录
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'compare')
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        # 加载数据
        H_data = load_H_data(epoch)
        A = load_original_data()
        
        # 获取邻居信息
        H_edge_weights = get_H_neighbors(H_data, node_k)
        original_G = get_original_neighbors(A, node_k)
        
        # 绘制并保存图形
        H_save_path = os.path.join(save_dir, f'H_{node_k}.png')
        original_save_path = os.path.join(save_dir, f'original_{node_k}.png')
        
        plot_H_graph(H_edge_weights, node_k, H_save_path)
        plot_original_graph(original_G, node_k, original_save_path)
        
        print(f"分析完成！图形已保存到 {save_dir} 目录")
        print(f"H数据邻居图: {H_save_path}")
        print(f"原始数据邻居图: {original_save_path}")
        
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, required=True, help='要分析的epoch编号')
    parser.add_argument('--node', type=int, required=True, help='要分析的节点编号')
    args = parser.parse_args()
    
    analyze_node(args.epoch, args.node) 