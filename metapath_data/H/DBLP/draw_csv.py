import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

def draw_training_curves(csv_path, save_dir):
    """绘制训练过程的loss和F1分数变化曲线"""
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    
    # 设置seaborn样式
    sns.set_style("whitegrid")
    
    # 读取CSV文件
    df = pd.read_csv(csv_path)
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 绘制loss曲线
    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['loss'], 'b-', linewidth=2)
    plt.title('Training Loss over Epochs', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'loss_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 绘制F1分数曲线
    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['train_f1'], 'r-', label='Train F1', linewidth=2)
    plt.plot(df['epoch'], df['val_f1'], 'g-', label='Validation F1', linewidth=2)
    plt.plot(df['epoch'], df['test_f1'], 'b-', label='Test F1', linewidth=2)
    plt.title('F1 Scores over Epochs', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'f1_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 打印一些统计信息
    print("\n=== Training Statistics ===")
    print(f"Best validation F1: {df['val_f1'].max():.4f} at epoch {df['val_f1'].idxmax() + 1}")
    print(f"Best test F1: {df['test_f1'].max():.4f} at epoch {df['test_f1'].idxmax() + 1}")
    print(f"Final loss: {df['loss'].iloc[-1]:.4f}")
    print(f"\nPlots have been saved to:")
    print(f"1. {os.path.join(save_dir, 'loss_curve.png')}")
    print(f"2. {os.path.join(save_dir, 'f1_curves.png')}")

if __name__ == '__main__':
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 设置文件路径
    csv_path = os.path.join(current_dir, 'exp.csv')
    
    # 绘制曲线
    draw_training_curves(csv_path, current_dir) 