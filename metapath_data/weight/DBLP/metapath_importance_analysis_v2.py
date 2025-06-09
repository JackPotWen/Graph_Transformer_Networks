import os
# 设置环境变量以解决OpenMP冲突
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

import pickle
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
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

def generate_all_metapaths(edge_types, max_length=5):
    """生成所有可能的metapath组合"""
    all_metapaths = []
    
    # 生成2到max_length长度的所有可能metapath
    for length in range(2, max_length + 1):
        # 生成所有可能的组合
        for path in product(edge_types, repeat=length-1):
            # 确保路径是有效的（相邻节点类型匹配）
            valid_path = True
            current_path = []
            
            # 检查第一个边类型
            if len(path[0]) != 2:  # 确保边类型长度为2
                continue
                
            # 构建路径并检查有效性
            current_path.append(path[0])
            for i in range(1, len(path)):
                if len(path[i]) != 2:  # 确保边类型长度为2
                    valid_path = False
                    break
                # 检查相邻边类型是否匹配
                if current_path[-1][1] != path[i][0]:
                    valid_path = False
                    break
                current_path.append(path[i])
            
            if valid_path and len(current_path) > 0:
                # 构建完整的metapath
                metapath = current_path[0][0]  # 第一个节点的类型
                for edge_type in current_path:
                    if len(edge_type) == 2:  # 再次确认边类型长度
                        metapath += edge_type[1]  # 添加每个边的目标节点类型
                all_metapaths.append(metapath)
    
    # 去重
    all_metapaths = list(set(all_metapaths))
    # 按长度排序
    all_metapaths.sort(key=len)
    
    return all_metapaths

def calculate_metapath_scores(weights):
    """计算metapath的重要性分数，按照论文的方法"""
    print("\n=== Metapath重要性计算 ===")
    
    # DBLP数据集的边类型定义
    edge_types = ['PA', 'AP', 'PC', 'CP', 'I']  # Paper-Author, Author-Paper, Paper-Conference, Conference-Paper, Identity
    
    # 获取第一个epoch的权重
    Ws = weights[0]
    
    # 生成所有可能的metapath
    metapaths = generate_all_metapaths(edge_types, max_length=5)
    print(f"生成了 {len(metapaths)} 个可能的metapath")
    print("示例metapath:", metapaths[:5])  # 打印前5个metapath作为示例
    
    # 存储所有metapath的分数
    metapath_scores = []
    
    # 分析每一层
    for layer_idx, W_list in enumerate(Ws):
        print(f"\n--- 第 {layer_idx + 1} 层分析 ---")
        
        # 每个层有两个权重矩阵W1和W2
        for w_idx, W in enumerate(W_list):
            W = np.array(W) if isinstance(W, list) else W
            print(f"权重矩阵 W{w_idx + 1} 形状: {W.shape}")
            
            # 对每个metapath计算分数
            for metapath in metapaths:
                # 将metapath分解为边类型序列
                edge_sequence = []
                for i in range(len(metapath)-1):
                    edge_type = metapath[i:i+2]
                    if edge_type in edge_types:
                        edge_sequence.append(edge_type)
                    else:
                        break
                
                # 如果所有边类型都有效，计算分数
                if len(edge_sequence) == len(metapath)-1:
                    # 计算每个channel的metapath分数
                    for ch_idx in range(W.shape[0]):
                        # 获取当前channel的权重
                        ch_weights = F.softmax(torch.from_numpy(W[ch_idx]), dim=0).numpy()
                        
                        # 计算metapath分数
                        score = 1.0
                        for edge_type in edge_sequence:
                            edge_idx = edge_types.index(edge_type)
                            score *= ch_weights[edge_idx]
                        
                        # 记录结果
                        metapath_scores.append({
                            'Layer': layer_idx + 1,
                            'Weight_Matrix': f'W{w_idx + 1}',
                            'Channel': ch_idx + 1,
                            'Metapath': metapath,
                            'Score': score
                        })
    
    # 转换为DataFrame
    df = pd.DataFrame(metapath_scores)
    
    # 保存到CSV
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'metapath_scores.csv')
    df.to_csv(save_path, index=False)
    print(f"\n已保存metapath分数到: {save_path}")
    
    return df

def find_top_metapaths(df, top_k=5):
    """找出最重要的metapath"""
    print(f"\n=== Top {top_k} 最重要的Metapath ===")
    
    # 按分数排序
    top_metapaths = df.sort_values('Score', ascending=False).head(top_k)
    
    print("\n排名\tLayer\tWeight_Matrix\tChannel\tMetapath\t\t分数")
    print("-" * 70)
    
    for i, (_, row) in enumerate(top_metapaths.iterrows()):
        print(f"{i+1:2d}\t{row['Layer']}\t{row['Weight_Matrix']}\t\t{row['Channel']}\t{row['Metapath']}\t\t{row['Score']:.6f}")
    
    return top_metapaths

def analyze_metapath_patterns(df):
    """分析metapath模式"""
    print("\n=== Metapath模式分析 ===")
    
    # 按metapath分组计算平均分数
    metapath_avg = df.groupby('Metapath')['Score'].mean().sort_values(ascending=False)
    
    print("\n按Metapath平均分数排序:")
    print("-" * 40)
    for metapath, avg_score in metapath_avg.items():
        print(f"{metapath}: {avg_score:.6f}")
    
    # 按层和权重矩阵分组计算平均分数
    layer_w_avg = df.groupby(['Layer', 'Weight_Matrix'])['Score'].mean().unstack()
    
    print("\n按层和权重矩阵的平均分数:")
    print("-" * 40)
    print(layer_w_avg)

if __name__ == '__main__':
    try:
        # 加载权重矩阵
        weights = load_weights('best_weights.pkl')
        
        # 计算metapath分数
        df = calculate_metapath_scores(weights)
        
        # 找出最重要的metapath
        top_metapaths = find_top_metapaths(df, top_k=5)
        
        # 分析metapath模式
        analyze_metapath_patterns(df)
        
    except FileNotFoundError as e:
        print(f"错误: {e}")
        print("\n请确保'best_weights.pkl'文件存在于以下目录:")
        print(os.path.dirname(os.path.abspath(__file__)))
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc() 
        