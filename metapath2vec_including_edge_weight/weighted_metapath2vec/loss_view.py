import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from pathlib import Path

def load_training_stats(stats_path):
    """加载训练统计信息
    
    参数:
        stats_path (str): training_stats.json文件路径
        
    返回:
        dict: 训练统计信息
    """
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    return stats

def plot_loss_curve(stats, save_path=None, show_plot=True):
    """绘制loss曲线
    
    参数:
        stats (dict): 训练统计信息
        save_path (str, optional): 保存图片的路径
        show_plot (bool): 是否显示图片
    """
    # 提取loss数据
    epoch_losses = stats['epoch_losses']
    epochs = range(1, len(epoch_losses) + 1)
    
    # 创建图形
    plt.figure(figsize=(12, 8))
    
    # 绘制loss曲线
    plt.plot(epochs, epoch_losses, 'b-', linewidth=2, label='Training Loss')
    
    # 找到loss最低点
    min_loss_idx = np.argmin(epoch_losses)
    min_loss_epoch = epochs[min_loss_idx]
    min_loss_value = epoch_losses[min_loss_idx]
    
    # 标注loss最低点
    plt.scatter(min_loss_epoch, min_loss_value, color='red', s=100, zorder=5, 
               label=f'Min Loss: {min_loss_value:.4f} (Epoch {min_loss_epoch})')
    
    # 设置图形属性
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Loss Curve', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    
    # 添加统计信息
    min_loss = min(epoch_losses)
    max_loss = max(epoch_losses)
    final_loss = epoch_losses[-1]
    
    info_text = f'Min Loss: {min_loss:.4f} (Epoch {min_loss_idx + 1})\nMax Loss: {max_loss:.4f}\nFinal Loss: {final_loss:.4f}\nTotal Epochs: {len(epoch_losses)}'
    
    plt.text(0.02, 0.98, info_text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
             fontsize=10)
    
    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Loss曲线已保存到: {save_path}")
    
    if show_plot:
        plt.show()
    
    plt.close()

def plot_multiple_loss_curves(stats_files, save_path=None, show_plot=True):
    """绘制多个loss曲线进行对比
    
    参数:
        stats_files (list): 多个training_stats.json文件路径的列表
        save_path (str, optional): 保存图片的路径
        show_plot (bool): 是否显示图片
    """
    plt.figure(figsize=(14, 10))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    for i, stats_path in enumerate(stats_files):
        try:
            stats = load_training_stats(stats_path)
            epoch_losses = stats['epoch_losses']
            epochs = range(1, len(epoch_losses) + 1)
            
            # 从文件名提取标识信息
            filename = Path(stats_path).stem
            label = filename.replace('training_stats_', '')
            
            color = colors[i % len(colors)]
            plt.plot(epochs, epoch_losses, color=color, linewidth=2, label=label)
            
        except Exception as e:
            print(f"加载文件 {stats_path} 时出错: {str(e)}")
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Loss Curves Comparison', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"对比图已保存到: {save_path}")
    
    if show_plot:
        plt.show()
    
    plt.close()

def plot_training_metrics(stats, save_path=None, show_plot=True):
    """绘制训练指标（loss、F1-score、时间）
    
    参数:
        stats (dict): 训练统计信息
        save_path (str, optional): 保存图片的路径
        show_plot (bool): 是否显示图片
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Training Metrics Overview', fontsize=16, fontweight='bold')
    
    epochs = range(1, len(stats['epoch_losses']) + 1)
    
    # 1. Loss曲线
    axes[0, 0].plot(epochs, stats['epoch_losses'], 'b-', linewidth=2)
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 训练时间
    axes[0, 1].plot(epochs, stats['epoch_times'], 'g-', linewidth=2)
    axes[0, 1].set_title('Training Time per Epoch')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Time (seconds)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. F1-score (如果有评估数据)
    if 'epoch_evaluations' in stats and stats['epoch_evaluations']:
        train_f1 = [eval['train_f1'] for eval in stats['epoch_evaluations']]
        val_f1 = [eval['val_f1'] for eval in stats['epoch_evaluations']]
        test_f1 = [eval['test_f1'] for eval in stats['epoch_evaluations']]
        
        axes[1, 0].plot(epochs, train_f1, 'b-', linewidth=2, label='Train F1')
        axes[1, 0].plot(epochs, val_f1, 'r-', linewidth=2, label='Val F1')
        axes[1, 0].plot(epochs, test_f1, 'g-', linewidth=2, label='Test F1')
        axes[1, 0].set_title('F1-Score')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1-Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 累积训练时间
    cumulative_time = np.cumsum(stats['epoch_times'])
    axes[1, 1].plot(epochs, cumulative_time, 'm-', linewidth=2)
    axes[1, 1].set_title('Cumulative Training Time')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Time (seconds)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"训练指标图已保存到: {save_path}")
    
    if show_plot:
        plt.show()
    
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='可视化训练loss曲线')
    parser.add_argument('--stats_path', type=str, 
                        default='/home/kuei-jan/github/Graph_Transformer_Networks/metapath2vec_including_edge_weight/weighted_metapath2vec/experiment_result_with_weighted_sample/training_stats_w200_l50_d128_c5_ns5.json',
                        help='training_stats.json文件路径')
    parser.add_argument('--save_path', type=str, 
                        default='/home/kuei-jan/github/Graph_Transformer_Networks/metapath2vec_including_edge_weight/weighted_metapath2vec/experiment_result_with_weighted_sample/loss_curve.png',
                        help='保存图片的路径')
    parser.add_argument('--plot_type', type=str, default='loss', 
                        choices=['loss', 'metrics', 'comparison'],
                        help='绘图类型: loss(仅loss曲线), metrics(完整指标), comparison(多文件对比)')
    parser.add_argument('--comparison_files', type=str, nargs='+',
                        help='用于对比的多个training_stats.json文件路径')
    parser.add_argument('--no_show', action='store_true',
                        help='不显示图片，仅保存')
    
    args = parser.parse_args()
    
    # 确保保存目录存在
    if args.save_path:
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    
    if args.plot_type == 'comparison':
        if not args.comparison_files:
            print("对比模式需要指定 --comparison_files 参数")
            return
        
        plot_multiple_loss_curves(args.comparison_files, args.save_path, not args.no_show)
    else:
        # 加载训练统计信息
        print(f"加载训练统计信息: {args.stats_path}")
        stats = load_training_stats(args.stats_path)
        
        # 打印基本信息
        print(f"总训练轮数: {len(stats['epoch_losses'])}")
        print(f"最终loss: {stats['epoch_losses'][-1]:.4f}")
        print(f"总训练时间: {stats.get('total_time', 'N/A')}秒")
        
        if args.plot_type == 'loss':
            plot_loss_curve(stats, args.save_path, not args.no_show)
        elif args.plot_type == 'metrics':
            plot_training_metrics(stats, args.save_path, not args.no_show)

if __name__ == '__main__':
    main() 