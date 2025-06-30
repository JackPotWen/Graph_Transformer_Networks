import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pickle
import os
import argparse
import torch
from mpl_toolkits.mplot3d import Axes3D

def load_embeddings(embedding_path):
    """加载节点嵌入
    
    参数:
        embedding_path (str): 嵌入文件路径
        
    返回:
        np.ndarray: 节点嵌入矩阵
    """
    try:
        # 使用pickle直接加载
        with open(embedding_path, 'rb') as f:
            data = pickle.load(f)
            if isinstance(data, torch.Tensor):
                return data.cpu().numpy()
            elif isinstance(data, np.ndarray):
                return data
            else:
                raise ValueError(f"不支持的数据类型: {type(data)}")
    except Exception as e:
        print(f"加载失败: {str(e)}")
        raise

def visualize_embeddings(embeddings, save_path=None, n_components=2):
    """使用PCA将节点嵌入可视化
    
    参数:
        embeddings (np.ndarray): 节点嵌入矩阵
        save_path (str, optional): 保存图片的路径
        n_components (int): PCA降维后的维度 (2或3)
    """
    # 定义节点类型和索引范围
    num_authors = 4057  # 0-4056
    num_papers = 14328  # 4057-18384
    num_conferences = 20  # 18385-18404
    
    # 使用PCA降维
    pca = PCA(n_components=n_components)
    embeddings_reduced = pca.fit_transform(embeddings)
    
    # 创建图形
    if n_components == 2:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
    else:  # n_components == 3
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
    
    # 绘制作者节点
    if n_components == 2:
        ax.scatter(embeddings_reduced[:num_authors, 0], 
                  embeddings_reduced[:num_authors, 1], 
                  c='red', 
                  label='Authors', 
                  alpha=0.6, 
                  s=10)
    else:
        ax.scatter(embeddings_reduced[:num_authors, 0], 
                  embeddings_reduced[:num_authors, 1],
                  embeddings_reduced[:num_authors, 2],
                  c='red', 
                  label='Authors', 
                  alpha=0.6, 
                  s=10)
    
    # 绘制论文节点
    if n_components == 2:
        ax.scatter(embeddings_reduced[num_authors:num_authors+num_papers, 0], 
                  embeddings_reduced[num_authors:num_authors+num_papers, 1], 
                  c='blue', 
                  label='Papers', 
                  alpha=0.4, 
                  s=5)
    else:
        ax.scatter(embeddings_reduced[num_authors:num_authors+num_papers, 0], 
                  embeddings_reduced[num_authors:num_authors+num_papers, 1],
                  embeddings_reduced[num_authors:num_authors+num_papers, 2],
                  c='blue', 
                  label='Papers', 
                  alpha=0.4, 
                  s=5)
    
    # 绘制会议节点
    if n_components == 2:
        ax.scatter(embeddings_reduced[num_authors+num_papers:, 0], 
                  embeddings_reduced[num_authors+num_papers:, 1], 
                  c='green', 
                  label='Conferences', 
                  alpha=0.8, 
                  s=20)
    else:
        ax.scatter(embeddings_reduced[num_authors+num_papers:, 0], 
                  embeddings_reduced[num_authors+num_papers:, 1],
                  embeddings_reduced[num_authors+num_papers:, 2],
                  c='green', 
                  label='Conferences', 
                  alpha=0.8, 
                  s=20)
    
    # 添加图例和标题
    ax.legend()
    ax.set_title(f'Node Embeddings Visualization (PCA - {n_components}D)')
    
    # 设置坐标轴标签
    if n_components == 2:
        ax.set_xlabel('First Principal Component')
        ax.set_ylabel('Second Principal Component')
    else:
        ax.set_xlabel('First Principal Component')
        ax.set_ylabel('Second Principal Component')
        ax.set_zlabel('Third Principal Component')
    
    # 添加解释方差比例信息
    explained_variance = pca.explained_variance_ratio_
    variance_text = 'Explained variance ratio:\n'
    for i, ratio in enumerate(explained_variance):
        variance_text += f'PC{i+1}: {ratio:.3f}\n'
    
    # 计算累积解释方差
    cumulative_variance = np.sum(explained_variance)
    variance_text += f'\nCumulative: {cumulative_variance:.3f}'
    
    # 添加节点数量信息
    nodes_text = f'Number of nodes:\nAuthors: {num_authors}\nPapers: {num_papers}\nConferences: {num_conferences}'
    
    # 在2D和3D图中使用不同的方式添加文本
    if n_components == 2:
        # 2D图使用plt.text
        plt.text(0.02, 0.98, variance_text,
                transform=plt.gca().transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        plt.text(0.02, 0.85, nodes_text,
                transform=plt.gca().transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        # 3D图使用fig.text
        fig.text(0.02, 0.98, variance_text,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        fig.text(0.02, 0.75, nodes_text,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"可视化结果已保存到: {save_path}")
    
    plt.show()

def main():
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='使用PCA进行节点嵌入可视化')
    parser.add_argument('--vec_feature_path', type=str, 
                        default='./metapath2vec_including_edge_weight/original_metapath2vec/experiment_result/vec_feature.pkl',
                        help='节点嵌入文件路径')
    parser.add_argument('--save_path', type=str, 
                        default='./metapath2vec_including_edge_weight/original_metapath2vec/experiment_result/node_embeddings_visualization_weighted_m2v.png',
                        help='输出图片保存路径')
    parser.add_argument('--n_components', type=int, default=2, choices=[2, 3],
                        help='PCA降维后的维度 (2或3)')
    
    args = parser.parse_args()
    
    # 确保保存目录存在
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    
    # 加载嵌入
    print("加载节点嵌入...")
    embeddings = load_embeddings(args.vec_feature_path)
    
    # 可视化
    print(f"生成{args.n_components}D可视化...")
    visualize_embeddings(embeddings, args.save_path, args.n_components)

if __name__ == '__main__':
    main() 