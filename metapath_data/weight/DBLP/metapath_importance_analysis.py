import os
# 设置环境变量以解决OpenMP冲突
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

import pickle
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product

def load_weights(file_path):
    """加载权重矩阵"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(current_dir, file_path)
    
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"找不到权重文件: {full_path}")
        
    with open(full_path, 'rb') as f:
        weights = pickle.load(f)
    return weights

def analyze_metapath_importance(weights):
    """分析metapath的重要性，类似于论文Table 3"""
    print("\n=== Metapath重要性分析 ===")
    
    # DBLP数据集的边类型定义
    edge_types = ['PA', 'AP', 'PC', 'CP', 'I']  # Paper-Author, Author-Paper, Paper-Conference, Conference-Paper, Identity
    
    # 获取第一个epoch的权重
    Ws = weights[0]
    
    print(f"总共 {len(Ws)} 层GTN")
    print(f"每层包含 {len(Ws[0])} 个权重矩阵")
    
    # 分析每一层的metapath重要性
    for layer_idx, W_list in enumerate(Ws):
        print(f"\n=== 第 {layer_idx + 1} 层分析 ===")
        
        for w_idx, W in enumerate(W_list):
            print(f"\n--- 权重矩阵 W{w_idx + 1} ---")
            
            # 确保W是numpy数组
            W = np.array(W) if isinstance(W, list) else W
            print(f"权重矩阵形状: {W.shape}")
            
            # 对每一行（每个channel）进行分析
            for row_idx in range(W.shape[0]):
                print(f"\nChannel {row_idx + 1}:")
                
                # 获取当前行的权重
                row_weights = W[row_idx]
                
                # 计算softmax概率
                row_probs = F.softmax(torch.from_numpy(row_weights), dim=0).numpy()
                
                # 打印每种边类型的重要性
                print("边类型重要性:")
                for i, (edge_type, prob) in enumerate(zip(edge_types, row_probs)):
                    print(f"  {edge_type}: {prob:.4f}")
                
                # 找出最重要的边类型
                max_idx = np.argmax(row_probs)
                print(f"最重要的边类型: {edge_types[max_idx]} ({row_probs[max_idx]:.4f})")

def calculate_metapath_scores(weights, max_length=3):
    """计算不同长度metapath的重要性分数"""
    print(f"\n=== Metapath分数计算 (最大长度: {max_length}) ===")
    
    edge_types = ['PA', 'AP', 'PC', 'CP', 'I']
    Ws = weights[0]
    
    # 存储所有metapath的分数
    metapath_scores = {}
    
    for layer_idx, W_list in enumerate(Ws):
        print(f"\n--- 第 {layer_idx + 1} 层 ---")
        
        for w_idx, W in enumerate(W_list):
            W = np.array(W) if isinstance(W, list) else W
            
            for row_idx in range(W.shape[0]):
                row_probs = F.softmax(torch.from_numpy(W[row_idx]), dim=0).numpy()
                
                # 计算不同长度的metapath分数
                for length in range(1, max_length + 1):
                    # 生成所有可能的metapath组合
                    metapaths = list(product(edge_types, repeat=length))
                    
                    for metapath in metapaths:
                        # 计算metapath的分数（简单乘法）
                        score = 1.0
                        for edge_type in metapath:
                            edge_idx = edge_types.index(edge_type)
                            score *= row_probs[edge_idx]
                        
                        metapath_str = '-'.join(metapath)
                        key = f"Layer{layer_idx+1}_W{w_idx+1}_Ch{row_idx+1}_{metapath_str}"
                        metapath_scores[key] = score
    
    return metapath_scores

def find_top_metapaths(metapath_scores, top_k=10):
    """找出最重要的metapath"""
    print(f"\n=== Top {top_k} 最重要的Metapath ===")
    
    # 按分数排序
    sorted_metapaths = sorted(metapath_scores.items(), key=lambda x: x[1], reverse=True)
    
    print("排名\tMetapath\t\t\t\t分数")
    print("-" * 60)
    
    for i, (metapath, score) in enumerate(sorted_metapaths[:top_k]):
        print(f"{i+1:2d}\t{metapath:<40}\t{score:.6f}")
    
    return sorted_metapaths[:top_k]

def analyze_metapath_patterns(metapath_scores):
    """分析metapath模式"""
    print("\n=== Metapath模式分析 ===")
    
    # 统计不同长度的metapath
    length_stats = {}
    edge_type_stats = {}
    
    for metapath_key, score in metapath_scores.items():
        # 提取metapath部分
        parts = metapath_key.split('_')
        metapath_str = parts[-1]
        edge_sequence = metapath_str.split('-')
        
        # 统计长度
        length = len(edge_sequence)
        if length not in length_stats:
            length_stats[length] = []
        length_stats[length].append(score)
        
        # 统计边类型
        for edge_type in edge_sequence:
            if edge_type not in edge_type_stats:
                edge_type_stats[edge_type] = []
            edge_type_stats[edge_type].append(score)
    
    print("\n按长度统计:")
    for length in sorted(length_stats.keys()):
        scores = length_stats[length]
        print(f"长度 {length}: 平均分数 {np.mean(scores):.6f}, 最高分数 {np.max(scores):.6f}")
    
    print("\n按边类型统计:")
    for edge_type in sorted(edge_type_stats.keys()):
        scores = edge_type_stats[edge_type]
        print(f"{edge_type}: 平均分数 {np.mean(scores):.6f}, 最高分数 {np.max(scores):.6f}")

def plot_metapath_importance_heatmap(weights):
    """绘制metapath重要性的热力图"""
    print("\n=== 生成Metapath重要性热力图 ===")
    
    edge_types = ['PA', 'AP', 'PC', 'CP', 'I']
    Ws = weights[0]
    
    # 创建保存目录
    save_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(save_dir, 'metapath_heatmaps'), exist_ok=True)
    
    for layer_idx, W_list in enumerate(Ws):
        for w_idx, W in enumerate(W_list):
            W = np.array(W) if isinstance(W, list) else W
            
            # 创建热力图
            plt.figure(figsize=(10, 6))
            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.rcParams['axes.unicode_minus'] = False
            
            # 对每一行进行softmax
            softmax_weights = []
            for row_idx in range(W.shape[0]):
                row_probs = F.softmax(torch.from_numpy(W[row_idx]), dim=0).numpy()
                softmax_weights.append(row_probs)
            
            softmax_weights = np.array(softmax_weights)
            
            # 绘制热力图
            sns.heatmap(softmax_weights, 
                       xticklabels=edge_types,
                       yticklabels=[f'Ch{i+1}' for i in range(W.shape[0])],
                       annot=True, 
                       fmt='.3f',
                       cmap='YlOrRd',
                       cbar_kws={'label': 'Attention Weight'})
            
            plt.title(f'Layer {layer_idx+1} W{w_idx+1} Metapath重要性')
            plt.xlabel('Edge Types')
            plt.ylabel('Channels')
            
            # 保存图片
            save_path = os.path.join(save_dir, 'metapath_heatmaps', 
                                   f'layer{layer_idx+1}_w{w_idx+1}_metapath_importance.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"已保存热力图: {save_path}")

if __name__ == '__main__':
    try:
        # 加载权重矩阵
        weights = load_weights('best_weights.pkl')
        
        # 分析metapath重要性
        analyze_metapath_importance(weights)
        
        # 计算metapath分数
        metapath_scores = calculate_metapath_scores(weights, max_length=3)
        
        # 找出最重要的metapath
        top_metapaths = find_top_metapaths(metapath_scores, top_k=15)
        
        # 分析metapath模式
        analyze_metapath_patterns(metapath_scores)
        
        # 绘制热力图
        plot_metapath_importance_heatmap(weights)
        
    except FileNotFoundError as e:
        print(f"错误: {e}")
        print("\n请确保'best_weights.pkl'文件存在于以下目录:")
        print(os.path.dirname(os.path.abspath(__file__)))
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc() 