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
        np.ndarray: 节点嵌入矩阵
    """
    try:
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

def load_labels(label_path):
    """加载标签数据
    
    参数:
        label_path (str): 标签文件路径
        
    返回:
        tuple: (train_labels, val_labels, test_labels)
    """
    with open(label_path, 'rb') as f:
        labels = pickle.load(f)
    
    # 合并所有标签
    all_labels = np.concatenate([labels[0], labels[1], labels[2]])
    # 按节点索引排序
    all_labels = all_labels[all_labels[:, 0].argsort()]
    return all_labels

def visualize_all_nodes(embeddings, labels, save_path=None):
    """可视化所有节点（作者和论文），作者节点按类别着色
    
    参数:
        embeddings (np.ndarray): 节点嵌入矩阵
        labels (np.ndarray): 标签数据
        save_path (str, optional): 保存图片的路径
    """
    # 定义节点类型和索引范围
    num_authors = 4057  # 0-4056
    num_papers = 14328  # 4057-18384
    
    # 使用PCA降维
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)
    
    # 创建图形
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    
    # 绘制论文节点（灰色）
    ax.scatter(embeddings_2d[num_authors:num_authors+num_papers, 0], 
              embeddings_2d[num_authors:num_authors+num_papers, 1], 
              c='gray', 
              label='Papers', 
              alpha=0.3, 
              s=5)
    
    # 获取作者节点的标签
    author_labels = labels[:, 1]  # 只取标签值
    unique_labels = np.unique(author_labels)
    
    # 为每个类别选择不同的颜色
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    # 绘制作者节点，按类别着色
    for i, label in enumerate(unique_labels):
        # 只对作者节点（前num_authors个节点）应用mask
        author_mask = author_labels == label
        ax.scatter(embeddings_2d[:num_authors][author_mask, 0], 
                  embeddings_2d[:num_authors][author_mask, 1], 
                  c=[colors[i]], 
                  label=f'Authors (Class {int(label)})', 
                  alpha=0.6, 
                  s=10)
    
    # 添加图例和标题
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_title('Author-Paper Embeddings Visualization (PCA)\nAuthors colored by class')
    ax.set_xlabel('First Principal Component')
    ax.set_ylabel('Second Principal Component')
    
    # 添加解释方差比例信息
    explained_variance = pca.explained_variance_ratio_
    variance_text = 'Explained variance ratio:\n'
    for i, ratio in enumerate(explained_variance):
        variance_text += f'PC{i+1}: {ratio:.3f}\n'
    cumulative_variance = np.sum(explained_variance)
    variance_text += f'\nCumulative: {cumulative_variance:.3f}'
    
    plt.text(0.02, 0.98, variance_text,
            transform=plt.gca().transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 调整布局以适应图例
    plt.tight_layout()
    
    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"可视化结果已保存到: {save_path}")
    
    plt.show()

def visualize_authors_only(embeddings, labels, save_path=None):
    """只可视化作者节点，按类别着色
    
    参数:
        embeddings (np.ndarray): 节点嵌入矩阵
        labels (np.ndarray): 标签数据
        save_path (str, optional): 保存图片的路径
    """
    # 只取作者节点的嵌入
    author_embeddings = embeddings[:4057]  # 0-4056是作者节点
    
    # 使用PCA降维
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(author_embeddings)
    
    # 创建图形
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    
    # 获取作者节点的标签
    author_labels = labels[:, 1]  # 只取标签值
    unique_labels = np.unique(author_labels)
    
    # 为每个类别选择不同的颜色
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    # 绘制作者节点，按类别着色
    for i, label in enumerate(unique_labels):
        mask = author_labels == label
        ax.scatter(embeddings_2d[mask, 0], 
                  embeddings_2d[mask, 1], 
                  c=[colors[i]], 
                  label=f'Class {int(label)}', 
                  alpha=0.6, 
                  s=10)
    
    # 添加图例和标题
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_title('Author Embeddings Visualization (PCA)\nColored by class')
    ax.set_xlabel('First Principal Component')
    ax.set_ylabel('Second Principal Component')
    
    # 添加解释方差比例信息
    explained_variance = pca.explained_variance_ratio_
    variance_text = 'Explained variance ratio:\n'
    for i, ratio in enumerate(explained_variance):
        variance_text += f'PC{i+1}: {ratio:.3f}\n'
    cumulative_variance = np.sum(explained_variance)
    variance_text += f'\nCumulative: {cumulative_variance:.3f}'
    
    # 添加类别分布信息
    class_dist = np.bincount(author_labels.astype(int))
    dist_text = 'Class distribution:\n'
    for label in unique_labels:
        dist_text += f'Class {int(label)}: {class_dist[int(label)]}\n'
    
    plt.text(0.02, 0.98, variance_text,
            transform=plt.gca().transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.text(0.02, 0.75, dist_text,
            transform=plt.gca().transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 调整布局以适应图例
    plt.tight_layout()
    
    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"可视化结果已保存到: {save_path}")
    
    plt.show()

def main():
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='使用PCA对带标签的作者节点进行可视化')
    parser.add_argument('--vec_feature_path', type=str, 
                        default='./metapath2vec_including_edge_weight/original_metapath2vec/experiment_result/vec_feature.pkl',
                        help='节点嵌入文件路径')
    parser.add_argument('--label_path', type=str, 
                        default='./data/DBLP/labels.pkl',
                        help='标签文件路径')
    parser.add_argument('--save_dir', type=str, 
                        default='./metapath2vec_including_edge_weight/original_metapath2vec/experiment_result',
                        help='输出目录路径')
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 加载数据
    print("加载节点嵌入...")
    embeddings = load_embeddings(args.vec_feature_path)
    
    print("加载标签数据...")
    labels = load_labels(args.label_path)
    
    # 生成两个可视化图
    print("生成作者-论文节点可视化...")
    all_nodes_path = os.path.join(args.save_dir, 'author_paper_class_visualization.png')
    visualize_all_nodes(embeddings, labels, all_nodes_path)
    
    print("生成作者节点聚类可视化...")
    authors_only_path = os.path.join(args.save_dir, 'author_class_visualization.png')
    visualize_authors_only(embeddings, labels, authors_only_path)

if __name__ == '__main__':
    main() 