import os
# 设置环境变量以解决OpenMP冲突
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

import pickle
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

def load_weights(file_path):
    """加载权重矩阵"""
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建完整的文件路径
    full_path = os.path.join(current_dir, file_path)
    
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"找不到权重文件: {full_path}")
        
    with open(full_path, 'rb') as f:
        weights = pickle.load(f)
    return weights

def analyze_weights(weights):
    """分析权重矩阵的形状和内容"""
    print("\n=== 权重矩阵分析 ===")
    # weights[0]是第一个（也是唯一的）epoch的权重
    Ws = weights[0]  # 获取第一个epoch的权重列表
    
    for i, W_list in enumerate(Ws):
        print(f"\nChannel {i+1}的权重矩阵:")
        for j, W in enumerate(W_list):
            # 确保W是numpy数组
            if isinstance(W, list):
                W = np.array(W)
            print(f"W{j+1} 形状: {W.shape}")
            print(f"W{j+1} 数值范围: [{W.min():.4f}, {W.max():.4f}]")
            print(f"W{j+1} 均值: {W.mean():.4f}")
            print(f"W{j+1} 标准差: {W.std():.4f}")
            print("-" * 50)

def plot_single_heatmap(scores, title, edge_types, save_path):
    """绘制单个热力图并保存"""
    plt.figure(figsize=(2, 8))  # 调整图形比例为竖直
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    
    # 创建热力图，转置数据使其竖直显示
    ax = sns.heatmap(scores.reshape(-1, 1),  # 转置数据
                    cmap='YlOrRd',
                    annot=False,  # 不显示数值
                    cbar=True,
                    square=False,
                    yticklabels=edge_types,  # 将边类型标签放在y轴
                    xticklabels=False,  # 不显示x轴标签
                    cbar_kws={'ticks': [],  # 不显示刻度值
                             'label': ''})  # 不显示标签
    
    plt.title(title, pad=20)
    plt.ylabel('Edge Types', labelpad=10)  # 将标签改为y轴
    
    # 调整布局
    plt.subplots_adjust(left=0.3)  # 为y轴标签留出空间
    
    # 保存图片
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存热力图: {save_path}")

def plot_attention_heatmap(weights):
    """绘制并保存每个attention score热力图"""
    # 获取第一个epoch的权重
    Ws = weights[0]
    
    # 创建保存目录
    save_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(save_dir, 'heatmaps'), exist_ok=True)
    
    # 对每个W矩阵的每一行进行softmax
    edge_types = ['PA', 'AP', 'PC', 'CP', 'I']
    plot_count = 0
    
    for i, W_list in enumerate(Ws):
        for j, W in enumerate(W_list):
            # 确保W是numpy数组
            W = np.array(W) if isinstance(W, list) else W
            # 对每一行分别进行softmax
            for row in range(W.shape[0]):
                row_softmax = F.softmax(torch.from_numpy(W[row]), dim=0).numpy()
                title = f'W{j+1} Row {row+1}'
                save_path = os.path.join(save_dir, 'heatmaps', f'heatmap_{plot_count+1}.png')
                plot_single_heatmap(row_softmax, title, edge_types, save_path)
                plot_count += 1

if __name__ == '__main__':
    try:
        # 加载权重矩阵
        weights = load_weights('best_weights.pkl')
        
        # 分析权重矩阵
        analyze_weights(weights)
        
        # 绘制attention score热力图
        plot_attention_heatmap(weights)
    except FileNotFoundError as e:
        print(f"错误: {e}")
        print("\n请确保'best_weights.pkl'文件存在于以下目录:")
        print(os.path.dirname(os.path.abspath(__file__)))
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()  # 打印详细的错误堆栈 