import numpy as np
import matplotlib.pyplot as plt
from umap.umap_ import UMAP
import pickle
import os
import argparse

def load_embeddings(embedding_path):
    """加载节点嵌入
    
    参数:
        embedding_path (str): 嵌入文件路径
        
    返回:
        np.ndarray: 节点嵌入矩阵
    """
    with open(embedding_path, 'rb') as f:
        embeddings = pickle.load(f)
    return embeddings

def visualize_embeddings(embeddings, save_path=None, n_neighbors=15, min_dist=0.1, metric='euclidean'):
    """使用UMAP将节点嵌入可视化
    
    参数:
        embeddings (np.ndarray): 节点嵌入矩阵
        save_path (str, optional): 保存图片的路径
        n_neighbors (int): UMAP的邻居数量参数
        min_dist (float): UMAP的最小距离参数
        metric (str): UMAP的距离度量方式
    """
    # 定义节点类型和索引范围
    num_authors = 4057  # 0-4056
    num_papers = 14328  # 4057-18384
    num_conferences = 20  # 18385-18404
    
    # 使用UMAP降维到2维
    reducer = UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=42
    )
    embeddings_2d = reducer.fit_transform(embeddings)
    
    # 创建图形
    plt.figure(figsize=(12, 8))
    
    # 绘制作者节点
    plt.scatter(embeddings_2d[:num_authors, 0], 
               embeddings_2d[:num_authors, 1], 
               c='red', 
               label='Authors', 
               alpha=0.6, 
               s=10)
    
    # 绘制论文节点
    plt.scatter(embeddings_2d[num_authors:num_authors+num_papers, 0], 
               embeddings_2d[num_authors:num_authors+num_papers, 1], 
               c='blue', 
               label='Papers', 
               alpha=0.4, 
               s=5)
    
    # 绘制会议节点
    plt.scatter(embeddings_2d[num_authors+num_papers:, 0], 
               embeddings_2d[num_authors+num_papers:, 1], 
               c='green', 
               label='Conferences', 
               alpha=0.8, 
               s=20)
    
    # 添加图例和标题
    plt.legend()
    plt.title(f'Node Embeddings Visualization (UMAP)\nn_neighbors={n_neighbors}, min_dist={min_dist}, metric={metric}')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    
    # 添加参数信息
    plt.text(0.02, 0.98, 
             f'UMAP Parameters:\nn_neighbors: {n_neighbors}\nmin_dist: {min_dist}\nmetric: {metric}',
             transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"可视化结果已保存到: {save_path}")
    
    plt.show()

def main():
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='使用UMAP进行节点嵌入可视化')
    parser.add_argument('--input', type=str, 
                        default='F:/github/Graph_Transformer_Networks/experiment_result/vec_feature.pkl',
                        help='输入嵌入文件路径')
    parser.add_argument('--output', type=str, 
                        default='F:/github/Graph_Transformer_Networks/experiment_result/node_embeddings_umap.png',
                        help='输出图片保存路径')
    parser.add_argument('--n_neighbors', type=int, default=15,
                        help='UMAP的邻居数量参数')
    parser.add_argument('--min_dist', type=float, default=0.1,
                        help='UMAP的最小距离参数')
    parser.add_argument('--metric', type=str, default='euclidean',
                        choices=['euclidean', 'manhattan', 'cosine', 'correlation'],
                        help='UMAP的距离度量方式')
    
    args = parser.parse_args()
    
    # 确保保存目录存在
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # 加载嵌入
    print("加载节点嵌入...")
    embeddings = load_embeddings(args.input)
    
    # 可视化
    print("生成UMAP可视化...")
    visualize_embeddings(
        embeddings, 
        save_path=args.output,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric
    )

if __name__ == '__main__':
    main() 