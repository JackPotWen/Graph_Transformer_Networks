import os
import pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
from analyze_node import load_H_data, load_original_data

def get_one_hop_neighbors(A, node_k):
    """获取节点k的一跳邻居"""
    one_hop_neighbors = set()
    for edge_type_idx, edge_type in enumerate(A):
        # 获取当前节点的邻居（排除自环）
        neighbors = edge_type[1][(edge_type[0] == node_k) & (edge_type[1] != node_k)]
        one_hop_neighbors.update(neighbors)
    return one_hop_neighbors

def find_paths_to_node(A, start_node, target_node, max_hops=3):
    """查找从start_node到target_node的所有可能路径"""
    paths = []
    visited = set([start_node])
    current_paths = {start_node: [[start_node]]}
    
    # 进行多跳遍历
    for hop in range(max_hops):
        next_paths = defaultdict(list)
        for node, node_paths in current_paths.items():
            # 遍历所有边类型
            for edge_type_idx, edge_type in enumerate(A):
                # 获取当前节点的邻居（排除自环）
                neighbors = edge_type[1][(edge_type[0] == node) & (edge_type[1] != node)]
                for neighbor in neighbors:
                    if neighbor not in visited:
                        # 为每个现有路径添加新节点
                        for path in node_paths:
                            new_path = path + [neighbor]
                            next_paths[neighbor].append(new_path)
                            # 如果到达目标节点，记录路径
                            if neighbor == target_node:
                                paths.append((new_path, edge_type_idx))
        current_paths = next_paths
        visited.update(current_paths.keys())
        
        # 如果已经找到目标节点，可以提前结束
        if target_node in visited:
            break
    
    return paths

def analyze_node_relations(H_data, A, node_k):
    """分析节点k的关系形成"""
    # 获取一跳邻居
    one_hop_neighbors = get_one_hop_neighbors(A, node_k)
    
    # 获取H数据中的边（排除自环）
    H_edges = set()
    edge_weights = defaultdict(float)
    edge_count = defaultdict(int)
    
    for channel_idx, channel_H in enumerate(H_data):
        edge_index, edge_value = channel_H
        for i, (src, dst) in enumerate(zip(edge_index[0], edge_index[1])):
            if src == node_k and dst != node_k:  # 排除自环
                edge = (src, dst)
                H_edges.add(edge)
                edge_weights[edge] += edge_value[i]
                edge_count[edge] += 1
    
    # 计算平均权重
    for edge in edge_weights:
        edge_weights[edge] /= edge_count[edge]
    
    # 分析不在一跳邻居中的节点
    analysis_results = {}
    for dst in set(edge[1] for edge in H_edges):
        if dst not in one_hop_neighbors:
            # 查找所有可能的路径
            paths = find_paths_to_node(A, node_k, dst)
            if paths:  # 如果存在路径
                analysis_results[(node_k, dst)] = {
                    'weight': edge_weights.get((node_k, dst), 0),
                    'paths': paths,
                    'is_one_hop': False
                }
        else:
            # 记录一跳邻居的边权重
            analysis_results[(node_k, dst)] = {
                'weight': edge_weights.get((node_k, dst), 0),
                'paths': [],  # 一跳邻居不需要路径
                'is_one_hop': True
            }
    
    return analysis_results, one_hop_neighbors

def get_edge_weights_in_H(H_data):
    """获取H数据中所有边的权重"""
    edge_weights = defaultdict(float)
    edge_count = defaultdict(int)
    
    for channel_H in H_data:
        edge_index, edge_value = channel_H
        for i, (src, dst) in enumerate(zip(edge_index[0], edge_index[1])):
            if src != dst:  # 排除自环
                edge = (src, dst)
                edge_weights[edge] += edge_value[i]
                edge_count[edge] += 1
    
    # 计算平均权重
    for edge in edge_weights:
        edge_weights[edge] /= edge_count[edge]
    
    return edge_weights

def plot_analysis_graph(analysis_results, one_hop_neighbors, node_k, H_data, save_path):
    """绘制分析结果图"""
    G = nx.DiGraph()
    
    # 获取H数据中所有边的权重
    H_edge_weights = get_edge_weights_in_H(H_data)
    
    # 添加所有相关节点和边
    all_nodes = set([node_k])
    
    for (src, dst), analysis in analysis_results.items():
        all_nodes.add(dst)
        if not analysis['is_one_hop']:  # 非一跳邻居
            # 添加直接边（如果存在）
            if (src, dst) in H_edge_weights:
                G.add_edge(src, dst, 
                          weight=H_edge_weights[(src, dst)], 
                          is_new=True)
            
            # 添加可能路径中的节点和边
            for path, _ in analysis['paths']:
                all_nodes.update(path)
                # 为路径中的每条边添加权重信息
                for i in range(len(path)-1):
                    edge = (path[i], path[i+1])
                    # 如果这条边在H中存在，添加权重
                    if edge in H_edge_weights:
                        G.add_edge(edge[0], edge[1], 
                                 weight=H_edge_weights[edge],
                                 is_path=True)
    
    plt.figure(figsize=(15, 10))
    pos = nx.spring_layout(G)
    
    # 绘制节点
    # 源节点
    nx.draw_networkx_nodes(G, pos, nodelist=[node_k], 
                          node_color='red', node_size=1000, 
                          label='Source Node')
    # 一跳邻居
    one_hop_nodes = [n for n in G.nodes() if n in one_hop_neighbors]
    nx.draw_networkx_nodes(G, pos, nodelist=one_hop_nodes,
                          node_color='lightgreen', node_size=500,
                          label='One-hop Neighbors')
    # 其他节点
    other_nodes = [n for n in G.nodes() if n != node_k and n not in one_hop_neighbors]
    nx.draw_networkx_nodes(G, pos, nodelist=other_nodes,
                          node_color='lightblue', node_size=500,
                          label='Multi-hop Nodes')
    
    # 绘制边
    new_edges = [(u, v) for (u, v, d) in G.edges(data=True) if d.get('is_new', False)]
    path_edges = [(u, v) for (u, v, d) in G.edges(data=True) if d.get('is_path', False)]
    
    nx.draw_networkx_edges(G, pos, edgelist=new_edges,
                          edge_color='red', width=2,
                          arrows=True, arrowsize=20,
                          label='Direct H Edges')
    nx.draw_networkx_edges(G, pos, edgelist=path_edges,
                          edge_color='gray', style='dashed',
                          arrows=True, arrowsize=15,
                          label='Path Edges in H')
    
    # 添加节点标签
    nx.draw_networkx_labels(G, pos, font_size=10)
    
    # 添加边权重标签
    # 1. 直接边的权重
    direct_edge_labels = {(u, v): f'{d["weight"]:.3f}' 
                         for u, v, d in G.edges(data=True) 
                         if d.get('is_new', False)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=direct_edge_labels,
                                font_size=8, font_color='red')
    
    # 2. 路径边的权重
    path_edge_labels = {(u, v): f'{d["weight"]:.3f}' 
                       for u, v, d in G.edges(data=True) 
                       if d.get('is_path', False)}
    
    # 调整路径边标签的位置，避免与直接边标签重叠
    path_pos = {}
    for edge in path_edge_labels:
        # 获取边的起点和终点位置
        start_pos = pos[edge[0]]
        end_pos = pos[edge[1]]
        # 计算边的中点，并稍微偏移
        mid_pos = (start_pos + end_pos) / 2
        # 添加随机偏移以避免重叠
        offset = np.array([np.random.uniform(-0.1, 0.1), 
                          np.random.uniform(-0.1, 0.1)])
        path_pos[edge] = mid_pos + offset
    
    # 绘制路径边标签
    for edge, label in path_edge_labels.items():
        plt.text(path_pos[edge][0], path_pos[edge][1], label,
                fontsize=8, color='gray',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
    plt.title(f'Analysis of Node {node_k} Relations in H Data\n'
             f'(Edge labels show actual weights in H)')
    plt.legend()
    plt.axis('off')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def generate_analysis_report(analysis_results, one_hop_neighbors, save_path):
    """生成分析报告"""
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("节点关系分析报告\n")
        f.write("=" * 50 + "\n\n")
        
        # 统计信息
        total_edges = len(analysis_results)
        one_hop_edges = sum(1 for analysis in analysis_results.values() if analysis['is_one_hop'])
        multi_hop_edges = total_edges - one_hop_edges
        
        f.write("统计信息:\n")
        f.write(f"- 总边数: {total_edges}\n")
        f.write(f"- 一跳邻居边数: {one_hop_edges}\n")
        f.write(f"- 多跳邻居边数: {multi_hop_edges}\n")
        f.write(f"- 一跳邻居节点数: {len(one_hop_neighbors)}\n\n")
        
        f.write("详细分析:\n")
        f.write("-" * 30 + "\n\n")
        
        # 分析一跳邻居
        f.write("一跳邻居分析:\n")
        for (src, dst), analysis in analysis_results.items():
            if analysis['is_one_hop']:
                f.write(f"边 ({src} -> {dst}):\n")
                f.write(f"- 权重: {analysis['weight']:.4f}\n")
                f.write(f"- 类型: 一跳邻居\n\n")
        
        # 分析多跳邻居
        f.write("\n多跳邻居分析:\n")
        for (src, dst), analysis in analysis_results.items():
            if not analysis['is_one_hop']:
                f.write(f"边 ({src} -> {dst}):\n")
                f.write(f"- 权重: {analysis['weight']:.4f}\n")
                f.write(f"- 类型: 多跳邻居\n")
                f.write("- 可能的路径:\n")
                
                # 按路径长度排序
                sorted_paths = sorted(analysis['paths'], key=lambda x: len(x[0]))
                for path, edge_type in sorted_paths:
                    path_str = " -> ".join(map(str, path))
                    f.write(f"  * 路径: {path_str}\n")
                    f.write(f"    长度: {len(path)-1}\n")
                    f.write(f"    边类型: {edge_type}\n")
                f.write("\n")
        
        # 添加路径统计
        f.write("\n路径统计:\n")
        path_lengths = []
        for analysis in analysis_results.values():
            if not analysis['is_one_hop']:
                for path, _ in analysis['paths']:
                    path_lengths.append(len(path)-1)
        
        if path_lengths:
            f.write(f"- 平均路径长度: {np.mean(path_lengths):.2f}\n")
            f.write(f"- 最短路径长度: {min(path_lengths)}\n")
            f.write(f"- 最长路径长度: {max(path_lengths)}\n")
            f.write(f"- 路径长度分布: {dict(zip(*np.unique(path_lengths, return_counts=True)))}\n")

def analyze_relations(epoch, node_k):
    """分析节点k在H数据中的关系形成"""
    # 创建保存目录
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'relation_analysis')
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        # 加载数据
        H_data = load_H_data(epoch)
        A = load_original_data()
        
        # 分析节点关系
        analysis_results, one_hop_neighbors = analyze_node_relations(H_data, A, node_k)
        
        # 绘制分析图
        graph_save_path = os.path.join(save_dir, f'relation_{node_k}.png')
        plot_analysis_graph(analysis_results, one_hop_neighbors, node_k, H_data, graph_save_path)
        
        # 生成分析报告
        report_save_path = os.path.join(save_dir, f'relation_{node_k}_report.txt')
        generate_analysis_report(analysis_results, one_hop_neighbors, report_save_path)
        
        print(f"分析完成！结果已保存到 {save_dir} 目录")
        print(f"分析图: {graph_save_path}")
        print(f"分析报告: {report_save_path}")
        
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
    
    analyze_relations(args.epoch, args.node) 