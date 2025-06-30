import torch
import numpy as np
import pickle
import argparse
from torch_geometric.utils import to_undirected
import sys
import os

# 添加上级目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from metapath2vec import MetaPath2Vec

import json
from tqdm import tqdm
import time
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

def load_acm_data(data_path):
    """加载 ACM 数据集
    
    参数:
        data_path (str): edges.pkl 文件路径
        
    返回:
        tuple: (edge_index_dict, num_nodes_dict)
    """
    with open(data_path, 'rb') as f:
        edges = pickle.load(f)
    
    # 定义节点类型和索引范围（根据之前的修正）
    num_papers = 3025      # 0-3024
    num_authors = 5912     # 3025-8936
    num_subjects = 57      # 8937-8993
    
    # 构建边索引字典
    edge_index_dict = {}
    
    # 处理 PA 边 (论文-作者)
    pa_matrix = edges[0]  # CSR格式
    pa_edges = pa_matrix.nonzero()
    edge_index_dict[('paper', 'written_by', 'author')] = torch.stack([
        torch.from_numpy(pa_edges[0].astype(np.int64)),  # 论文索引
        torch.from_numpy(pa_edges[1].astype(np.int64))   # 作者索引
    ])
    
    # 处理 AP 边 (作者-论文)
    ap_matrix = edges[1]  # CSC格式
    ap_edges = ap_matrix.nonzero()
    edge_index_dict[('author', 'writes', 'paper')] = torch.stack([
        torch.from_numpy(ap_edges[0].astype(np.int64)),  # 作者索引
        torch.from_numpy(ap_edges[1].astype(np.int64))   # 论文索引
    ])
    
    # 处理 PS 边 (论文-主题)
    ps_matrix = edges[2]  # CSR格式
    ps_edges = ps_matrix.nonzero()
    edge_index_dict[('paper', 'belongs_to', 'subject')] = torch.stack([
        torch.from_numpy(ps_edges[0].astype(np.int64)),  # 论文索引
        torch.from_numpy(ps_edges[1].astype(np.int64))   # 主题索引
    ])
    
    # 处理 SP 边 (主题-论文)
    sp_matrix = edges[3]  # CSC格式
    sp_edges = sp_matrix.nonzero()
    edge_index_dict[('subject', 'contains', 'paper')] = torch.stack([
        torch.from_numpy(sp_edges[0].astype(np.int64)),  # 主题索引
        torch.from_numpy(sp_edges[1].astype(np.int64))   # 论文索引
    ])
    
    # 构建节点数量字典
    num_nodes_dict = {
        'paper': num_papers,
        'author': num_authors,
        'subject': num_subjects
    }
    
    # 打印一些统计信息以验证数据加载
    print("数据加载统计信息:")
    print(f"论文节点数量: {num_papers}")
    print(f"作者节点数量: {num_authors}")
    print(f"主题节点数量: {num_subjects}")
    for edge_type, edge_index in edge_index_dict.items():
        print(f"边类型 {edge_type}: {edge_index.shape[1]} 条边")
    
    return edge_index_dict, num_nodes_dict

def load_labels(label_path):
    """加载标签数据
    
    参数:
        label_path (str): labels.pkl 文件路径
        
    返回:
        tuple: (train_labels, val_labels, test_labels)
    """
    with open(label_path, 'rb') as f:
        labels = pickle.load(f)
    
    # 转换为numpy数组
    train_labels = np.array(labels[0])  # shape: (N, 2) - (node_idx, label)
    val_labels = np.array(labels[1])
    test_labels = np.array(labels[2])
    
    return train_labels, val_labels, test_labels

def evaluate_embeddings(embeddings, train_labels, val_labels, test_labels):
    """使用逻辑回归评估嵌入质量
    
    参数:
        embeddings (np.ndarray): 节点嵌入
        train_labels (np.ndarray): 训练集标签
        val_labels (np.ndarray): 验证集标签
        test_labels (np.ndarray): 测试集标签
        
    返回:
        dict: 评估结果
    """
    # 只使用论文节点的嵌入和标签（ACM数据集中论文节点在前3025个）
    paper_embeddings = embeddings[:3025]  # 前3025个节点是论文
    
    # 准备训练数据
    X_train = paper_embeddings[train_labels[:, 0]]
    y_train = train_labels[:, 1]
    
    # 准备验证数据
    X_val = paper_embeddings[val_labels[:, 0]]
    y_val = val_labels[:, 1]
    
    # 准备测试数据
    X_test = paper_embeddings[test_labels[:, 0]]
    y_test = test_labels[:, 1]
    
    # 训练逻辑回归模型
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    
    # 预测并计算F1分数
    train_f1 = f1_score(y_train, clf.predict(X_train), average='micro')
    val_f1 = f1_score(y_val, clf.predict(X_val), average='micro')
    test_f1 = f1_score(y_test, clf.predict(X_test), average='micro')
    
    return {
        'train_f1': float(train_f1),
        'val_f1': float(val_f1),
        'test_f1': float(test_f1)
    }

def train_metapath2vec(args):
    """训练 MetaPath2Vec 模型
    
    参数:
        args: 命令行参数
    """
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 设置内存优化
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # 设置较小的内存分配
        torch.cuda.set_per_process_memory_fraction(0.8)
    
    # 加载数据
    print("正在加载数据...")
    edge_index_dict, num_nodes_dict = load_acm_data(args.data_path)
    
    # 加载标签数据
    print("加载标签数据...")
    train_labels, val_labels, test_labels = load_labels(args.label_path)
    
    # 定义元路径 PAP   (Paper-Author-Paper)
    metapath = [
    ('paper', 'written_by', 'author'),
    ('author', 'writes', 'paper')
    ]   
    
    # 自动判断元路径中涉及的节点类型
    def get_metapath_node_types(metapath):
        """从元路径中提取涉及的节点类型"""
        node_types = set()
        for edge_type in metapath:
            node_types.add(edge_type[0])  # 源节点类型
            node_types.add(edge_type[2])  # 目标节点类型
        return sorted(list(node_types))
    
    metapath_node_types = get_metapath_node_types(metapath)
    print(f"元路径涉及的节点类型: {metapath_node_types}")
    
    # 初始化模型 - 添加错误处理
    print("正在初始化模型...")
    try:
        model = MetaPath2Vec(
            edge_index_dict=edge_index_dict,
            embedding_dim=args.embedding_dim,
            metapath=metapath,
            walk_length=args.walk_length,
            context_size=args.context_size,
            walks_per_node=args.walks_per_node,
            num_negative_samples=args.num_negative_samples,
            num_nodes_dict=num_nodes_dict,
            sparse=True
        )
        
        # 延迟移动到GPU
        model = model.to(device)
        print("模型初始化成功！")
        
    except Exception as e:
        print(f"模型初始化失败: {str(e)}")
        print("尝试使用CPU模式...")
        device = torch.device('cpu')
        model = MetaPath2Vec(
            edge_index_dict=edge_index_dict,
            embedding_dim=args.embedding_dim,
            metapath=metapath,
            walk_length=args.walk_length,
            context_size=args.context_size,
            walks_per_node=args.walks_per_node,
            num_negative_samples=args.num_negative_samples,
            num_nodes_dict=num_nodes_dict,
            sparse=True
        ).to(device)
    
    # 设置优化器
    optimizer = torch.optim.SparseAdam(model.parameters(), lr=args.lr)
    
    # 用于记录训练过程
    training_stats = {
        'epoch_losses': [],
        'epoch_times': [],
        'epoch_evaluations': [],
        'total_time': 0,
        'parameters': vars(args),
        'metapath_node_types': metapath_node_types
    }
    
    # 训练模型
    print("\n开始训练...")
    model.train()
    start_time = time.time()
    
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        total_loss = 0
        num_batches = 0
        
        # 使用tqdm创建进度条
        pbar = tqdm(model.loader(batch_size=args.batch_size), 
                   desc=f'Epoch {epoch+1}/{args.epochs}',
                   leave=False)
        
        for pos_rw, neg_rw in pbar:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # 更新进度条
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # 计算平均损失
        avg_loss = total_loss / num_batches
        epoch_time = time.time() - epoch_start_time
        
        # 记录训练统计信息
        training_stats['epoch_losses'].append(float(avg_loss))
        training_stats['epoch_times'].append(float(epoch_time))
        
        # 在每个epoch结束后评估嵌入质量
        model.eval()
        with torch.no_grad():
            # 动态获取元路径中涉及的节点嵌入
            node_embeddings = []
            for node_type in metapath_node_types:
                node_emb = model(node_type).cpu()
                node_embeddings.append(node_emb)
            
            # 合并所有节点嵌入
            node_emb = torch.cat(node_embeddings, dim=0)
            
            # 评估嵌入质量
            eval_results = evaluate_embeddings(
                node_emb.numpy(), train_labels, val_labels, test_labels
            )
            training_stats['epoch_evaluations'].append(eval_results)
        
        model.train()  # 切换回训练模式
        
        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 打印训练和评估信息
        if (epoch + 1) % args.log_steps == 0:
            print(f'Epoch: {epoch+1:03d}/{args.epochs} | '
                  f'Loss: {avg_loss:.4f} | '
                  f'Time: {epoch_time:.2f}s | '
                  f'Train F1: {eval_results["train_f1"]:.4f} | '
                  f'Val F1: {eval_results["val_f1"]:.4f} | '
                  f'Test F1: {eval_results["test_f1"]:.4f}')
    
    # 计算总训练时间
    training_stats['total_time'] = time.time() - start_time
    
    # 生成元路径简写用于文件名
    def get_metapath_abbreviation(metapath):
        """从元路径生成简写"""
        abbreviation = ""
        for edge_type in metapath:
            # 取源节点类型的第一个字母
            abbreviation += edge_type[0][0].upper()
        # 添加最后一个目标节点类型的第一个字母
        abbreviation += metapath[-1][2][0].upper()
        return abbreviation.lower()
    
    metapath_abbr = get_metapath_abbreviation(metapath)
    print(f"元路径简写: {metapath_abbr}")
    
    # 保存最终嵌入
    print("\n保存最终节点嵌入...")
    model.eval()
    with torch.no_grad():
        # 动态获取元路径中涉及的节点嵌入
        node_embeddings = []
        for node_type in metapath_node_types:
            node_emb = model(node_type).cpu()
            node_embeddings.append(node_emb)
        
        # 合并所有节点嵌入
        node_emb = torch.cat(node_embeddings, dim=0)
        
        # 使用元路径简写命名文件
        save_path = os.path.join(args.save_dir, f'vec_feature_{metapath_abbr}.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(node_emb.numpy(), f)
        print(f'节点嵌入已保存到: {save_path}')
    
    # 保存训练统计信息
    stats_path = os.path.join(args.save_dir, f'training_stats_{metapath_abbr}.json')
    with open(stats_path, 'w') as f:
        json.dump(training_stats, f, indent=4)
    print(f'\n训练统计信息已保存到: {stats_path}')
    
    # 找出最佳epoch
    best_epoch = np.argmax([eval['val_f1'] for eval in training_stats['epoch_evaluations']])
    best_eval = training_stats['epoch_evaluations'][best_epoch]
    
    print("\n训练完成！")
    print(f"总训练时间: {training_stats['total_time']:.2f}秒")
    print(f"最佳epoch: {best_epoch + 1}")
    print(f"最佳验证集F1-score: {best_eval['val_f1']:.4f}")
    print(f"对应测试集F1-score: {best_eval['test_f1']:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/ACM/edges.pkl',
                        help='ACM 数据集路径')
    parser.add_argument('--save_dir', type=str, default='/home/kuei-jan/github/Graph_Transformer_Networks/metapath2vec_including_edge_weight/original_metapath2vec/ACM',
                        help='保存结果的目录')
    parser.add_argument('--embedding_dim', type=int, default=256,
                        help='嵌入维度')
    parser.add_argument('--walk_length', type=int, default=200,
                        help='随机游走长度')
    parser.add_argument('--context_size', type=int, default=10,
                        help='上下文窗口大小')
    parser.add_argument('--walks_per_node', type=int, default=200,
                        help='每个节点的游走次数')
    parser.add_argument('--num_negative_samples', type=int, default=5,
                        help='负样本数量')
    parser.add_argument('--epochs', type=int, default=50,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='批处理大小')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='学习率')
    parser.add_argument('--log_steps', type=int, default=1,
                        help='打印日志的步数')
    parser.add_argument('--label_path', type=str, 
                        default='data/ACM/labels.pkl',
                        help='标签数据路径')
    
    args = parser.parse_args()
    
    # 确保保存目录存在
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 训练模型
    train_metapath2vec(args) 