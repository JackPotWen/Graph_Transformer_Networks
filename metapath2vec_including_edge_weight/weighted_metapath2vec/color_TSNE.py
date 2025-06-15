import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pickle
import os
import argparse
import torch
import seaborn as sns

def load_embeddings(embedding_path):
    """加载节点嵌入
    
    参数:
        embedding_path (str): 嵌入文件路径
        
    返回:
        np.ndarray: 节点嵌入矩阵
    """
    try:
        # 使用torch.load时设置weights_only=False
        embeddings = torch.load(embedding_path, weights_only=False)
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()
        return embeddings
    except Exception as e:
        print(f"使用torch.load加载失败: {str(e)}")
        print("尝试使用pickle加载...")
        try:
            # 尝试使用pickle直接加载numpy数组
            with open(embedding_path, 'rb') as f:
                data = pickle.load(f)
                if isinstance(data, np.ndarray):
                    return data
                elif isinstance(data, torch.Tensor):
                    return data.cpu().numpy()
                else:
                    raise ValueError(f"不支持的数据类型: {type(data)}")
        except Exception as e:
            print(f"使用pickle加载失败: {str(e)}")
            raise

def load_labels(label_path):
    """加载标签数据
    
    参数:
        label_path (str): 标签文件路径
        
    返回:
        np.ndarray: 作者节点的标签
    """
    with open(label_path, 'rb') as f:
        train_labels, val_labels, test_labels = pickle.load(f)
    # 合并所有标签
    all_labels = np.concatenate([train_labels, val_labels, test_labels])
    return all_labels

def visualize_embeddings(embeddings, labels=None, save_path=None, perplexity=30, n_iter=1000, 
                        learning_rate='auto', color_by_class=False):
    """使用t-SNE将节点嵌入可视化
    
    参数:
        embeddings (np.ndarray): 节点嵌入矩阵
        labels (np.ndarray, optional): 作者节点的标签数据
        save_path (str, optional): 保存图片的路径
        perplexity (int): t-SNE的困惑度参数
        n_iter (int): 迭代次数
        learning_rate (str/float): 学习率
        color_by_class (bool): 是否按类别对作者节点进行染色
    """
    # 定义节点类型和索引范围
    num_authors = 4057  # 0-4056
    num_papers = 14328  # 4057-18384
    num_conferences = 20  # 18385-18404
    
    print("正在使用t-SNE进行降维...")
    # 使用t-SNE降维到2维
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=n_iter,
        learning_rate=learning_rate,
        random_state=42,
        verbose=1
    )
    embeddings_2d = tsne.fit_transform(embeddings)
    
    print("正在生成可视化...")
    # 创建图形
    plt.figure(figsize=(12, 8))
    
    if color_by_class and labels is not None:
        # 按类别对作者节点进行染色
        unique_classes = np.unique(labels[:, 1])
        palette = sns.color_palette('tab10', n_colors=len(unique_classes))
        
        # 绘制作者节点（按类别染色）
        for i, class_label in enumerate(unique_classes):
            mask = labels[:, 1] == class_label
            author_indices = labels[mask, 0]
            plt.scatter(embeddings_2d[author_indices, 0],
                       embeddings_2d[author_indices, 1],
                       c=[palette[i]],
                       label=f'Author (Class {class_label})',
                       alpha=0.6,
                       s=10)
    else:
        # 所有作者节点使用相同颜色
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
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    title = 'Node Embeddings Visualization (t-SNE)\n'
    title += f'perplexity={perplexity}, n_iter={n_iter}, learning_rate={learning_rate}'
    if color_by_class:
        title += '\nAuthors colored by class'
    plt.title(title)
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    
    # 添加参数信息
    plt.text(0.02, 0.98,
             f't-SNE Parameters:\nperplexity: {perplexity}\nn_iter: {n_iter}\nlearning_rate: {learning_rate}',
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
    parser = argparse.ArgumentParser(description='使用t-SNE进行节点嵌入可视化')
    parser.add_argument('--input', type=str,
                        default='F:/github/Graph_Transformer_Networks/experiment_result/vec_feature_m2v_woweight.pkl',
                        help='输入嵌入文件路径')
    parser.add_argument('--label_path', type=str,
                        default='F:/github/Graph_Transformer_Networks/data/DBLP/labels.pkl',
                        help='标签文件路径')
    parser.add_argument('--output', type=str,
                        default='F:/github/Graph_Transformer_Networks/experiment_result/node_embeddings_tsne_wo_m2v.png',
                        help='输出图片保存路径')
    parser.add_argument('--perplexity', type=int, default=40,
                        help='t-SNE的困惑度参数')
    parser.add_argument('--n_iter', type=int, default=2000,
                        help='t-SNE的迭代次数')
    parser.add_argument('--learning_rate', type=str, default='auto',
                        help='t-SNE的学习率 (auto 或大于0的浮点数)')
    parser.add_argument('--color_by_class', action='store_true',
                        help='是否按类别对作者节点进行染色')
    
    args = parser.parse_args()
    
    # 处理学习率参数
    try:
        if args.learning_rate.lower() != 'auto':
            learning_rate = float(args.learning_rate)
            if learning_rate <= 0:
                raise ValueError("学习率必须大于0")
        else:
            learning_rate = 'auto'
    except ValueError as e:
        print(f"学习率参数错误: {str(e)}")
        print("使用默认值 'auto'")
        learning_rate = 'auto'
    
    # 确保保存目录存在
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # 加载嵌入
    print("加载节点嵌入...")
    embeddings = load_embeddings(args.input)
    
    # 如果需要按类别染色，加载标签数据
    labels = None
    if args.color_by_class:
        print("加载标签数据...")
        labels = load_labels(args.label_path)
    
    # 可视化
    visualize_embeddings(
        embeddings,
        labels=labels,
        save_path=args.output,
        perplexity=args.perplexity,
        n_iter=args.n_iter,
        learning_rate=learning_rate,
        color_by_class=args.color_by_class
    )

if __name__ == '__main__':
    main() 