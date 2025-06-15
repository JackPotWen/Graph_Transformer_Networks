import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pickle
import os
import argparse
import torch

def load_embeddings(embedding_path):
    """加载节点嵌入
    
    参数:
        embedding_path (str): 嵌入文件路径
        
    返回:
        tuple: (author_embeddings, paper_embeddings)
    """
    try:
        # 使用torch.load时设置weights_only=False
        embeddings = torch.load(embedding_path, weights_only=False)
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()
    except Exception as e:
        print(f"使用torch.load加载失败: {str(e)}")
        print("尝试使用pickle加载...")
        try:
            with open(embedding_path, 'rb') as f:
                embeddings = pickle.load(f)
                if isinstance(embeddings, torch.Tensor):
                    embeddings = embeddings.cpu().numpy()
        except Exception as e:
            print(f"使用pickle加载失败: {str(e)}")
            raise
    
    # 分离作者和论文的嵌入
    num_authors = 4057  # 0-4056
    num_papers = 14328  # 4057-18384
    
    author_embeddings = embeddings[:num_authors]
    paper_embeddings = embeddings[num_authors:num_authors+num_papers]
    
    return author_embeddings, paper_embeddings

def visualize_embeddings(author_embeddings, paper_embeddings, save_path=None, n_components=2):
    """使用PCA将作者和论文节点嵌入可视化
    
    参数:
        author_embeddings (np.ndarray): 作者节点嵌入矩阵
        paper_embeddings (np.ndarray): 论文节点嵌入矩阵
        save_path (str, optional): 保存图片的路径
        n_components (int): PCA降维后的维度
    """
    # 合并作者和论文的嵌入用于PCA
    combined_embeddings = np.vstack([author_embeddings, paper_embeddings])
    
    # 使用PCA降维
    pca = PCA(n_components=n_components)
    embeddings_2d = pca.fit_transform(combined_embeddings)
    
    # 分离降维后的作者和论文嵌入
    author_2d = embeddings_2d[:len(author_embeddings)]
    paper_2d = embeddings_2d[len(author_embeddings):]
    
    # 创建图形
    plt.figure(figsize=(12, 8))
    
    # 绘制作者节点
    plt.scatter(author_2d[:, 0], 
               author_2d[:, 1], 
               c='red', 
               label='Authors', 
               alpha=0.6, 
               s=10)
    
    # 绘制论文节点
    plt.scatter(paper_2d[:, 0], 
               paper_2d[:, 1], 
               c='blue', 
               label='Papers', 
               alpha=0.4, 
               s=5)
    
    # 添加图例和标题
    plt.legend()
    plt.title('Author-Paper Embeddings Visualization (PCA)')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    
    # 添加解释方差比例信息
    explained_variance = pca.explained_variance_ratio_
    plt.text(0.02, 0.98, 
             f'Explained variance ratio:\nPC1: {explained_variance[0]:.3f}\nPC2: {explained_variance[1]:.3f}',
             transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 添加节点数量信息
    plt.text(0.02, 0.85,
             f'Number of nodes:\nAuthors: {len(author_embeddings)}\nPapers: {len(paper_embeddings)}',
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
    parser = argparse.ArgumentParser(description='使用PCA对作者和论文节点嵌入进行可视化')
    parser.add_argument('--input', type=str, 
                        default='F:/github/Graph_Transformer_Networks/metapath2vec_including_edge_weight/original_metapath2vec/experiment_result/vec_feature.pkl',
                        help='输入嵌入文件路径')
    parser.add_argument('--output', type=str, 
                        default='F:/github/Graph_Transformer_Networks/metapath2vec_including_edge_weight/original_metapath2vec/experiment_result/author_paper_visualization_weighted_m2v.png',
                        help='输出图片保存路径')
    
    args = parser.parse_args()
    
    # 确保保存目录存在
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # 加载嵌入
    print("加载节点嵌入...")
    author_embeddings, paper_embeddings = load_embeddings(args.input)
    
    # 可视化
    print("生成可视化...")
    visualize_embeddings(author_embeddings, paper_embeddings, args.output)

if __name__ == '__main__':
    main() 