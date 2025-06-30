"""
Training Accuracy and F1-score Visualization Tool
For plotting accuracy and F1-score curves during training process
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
from pathlib import Path

def plot_accuracy_curves(csv_file, save_dir=None):
    """
    Plot accuracy curves
    
    Args:
        csv_file: CSV file path
        save_dir: Directory to save images
    """
    # Read CSV file
    df = pd.read_csv(csv_file)
    
    # Set English font
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot accuracy curves
    ax1.plot(df['epoch'], df['train_acc'], label='Training Accuracy', linewidth=2, color='blue')
    ax1.plot(df['epoch'], df['val_acc'], label='Validation Accuracy', linewidth=2, color='red')
    ax1.plot(df['epoch'], df['test_acc'], label='Test Accuracy', linewidth=2, color='green')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Training, Validation and Test Accuracy Curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # Plot F1-score curves
    ax2.plot(df['epoch'], df['train_f1'], label='Training F1-score', linewidth=2, color='blue')
    ax2.plot(df['epoch'], df['val_f1'], label='Validation F1-score', linewidth=2, color='red')
    ax2.plot(df['epoch'], df['test_f1'], label='Test F1-score', linewidth=2, color='green')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('F1-Score')
    ax2.set_title('Training, Validation and Test F1-score Curves')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    
    # Save image
    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, 'accuracy_curves.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Accuracy curves saved to: {save_path}")
    
    plt.show()

def plot_accuracy_statistics(csv_file, save_dir=None):
    """
    Plot accuracy statistics
    
    Args:
        csv_file: CSV file path
        save_dir: Directory to save images
    """
    # Read CSV file
    df = pd.read_csv(csv_file)
    
    # Set English font
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Accuracy distribution histogram
    axes[0, 0].hist(df['train_acc'], bins=20, alpha=0.7, label='Training Accuracy', color='blue')
    axes[0, 0].hist(df['val_acc'], bins=20, alpha=0.7, label='Validation Accuracy', color='red')
    axes[0, 0].hist(df['test_acc'], bins=20, alpha=0.7, label='Test Accuracy', color='green')
    axes[0, 0].set_xlabel('Accuracy')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Accuracy Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. F1-score distribution histogram
    axes[0, 1].hist(df['train_f1'], bins=20, alpha=0.7, label='Training F1-score', color='blue')
    axes[0, 1].hist(df['val_f1'], bins=20, alpha=0.7, label='Validation F1-score', color='red')
    axes[0, 1].hist(df['test_f1'], bins=20, alpha=0.7, label='Test F1-score', color='green')
    axes[0, 1].set_xlabel('F1-Score')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('F1-score Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Accuracy boxplot
    acc_data = [df['train_acc'], df['val_acc'], df['test_acc']]
    axes[0, 2].boxplot(acc_data, labels=['Training Accuracy', 'Validation Accuracy', 'Test Accuracy'])
    axes[0, 2].set_ylabel('Accuracy')
    axes[0, 2].set_title('Accuracy Boxplot')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. F1-score boxplot
    f1_data = [df['train_f1'], df['val_f1'], df['test_f1']]
    axes[1, 0].boxplot(f1_data, labels=['Training F1-score', 'Validation F1-score', 'Test F1-score'])
    axes[1, 0].set_ylabel('F1-Score')
    axes[1, 0].set_title('F1-score Boxplot')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Accuracy vs F1-score scatter plot
    axes[1, 1].scatter(df['val_acc'], df['val_f1'], alpha=0.6, color='red', label='Validation Set')
    axes[1, 1].scatter(df['test_acc'], df['test_f1'], alpha=0.6, color='green', label='Test Set')
    axes[1, 1].set_xlabel('Accuracy')
    axes[1, 1].set_ylabel('F1-Score')
    axes[1, 1].set_title('Accuracy vs F1-score')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Training vs Validation accuracy scatter plot
    axes[1, 2].scatter(df['train_acc'], df['val_acc'], alpha=0.6, color='purple')
    axes[1, 2].set_xlabel('Training Accuracy')
    axes[1, 2].set_ylabel('Validation Accuracy')
    axes[1, 2].set_title('Training vs Validation Accuracy')
    axes[1, 2].grid(True, alpha=0.3)
    
    # Add correlation coefficient
    corr = df['train_acc'].corr(df['val_acc'])
    axes[1, 2].text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=axes[1, 2].transAxes, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    plt.tight_layout()
    
    # Save image
    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, 'accuracy_statistics.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Accuracy statistics saved to: {save_path}")
    
    plt.show()

def plot_performance_summary(csv_file, save_dir=None):
    """
    Plot performance summary
    
    Args:
        csv_file: CSV file path
        save_dir: Directory to save images
    """
    # Read CSV file
    df = pd.read_csv(csv_file)
    
    # Set English font
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Final performance comparison
    final_metrics = {
        'Training Accuracy': df['train_acc'].iloc[-1],
        'Validation Accuracy': df['val_acc'].iloc[-1],
        'Test Accuracy': df['test_acc'].iloc[-1],
        'Training F1-score': df['train_f1'].iloc[-1],
        'Validation F1-score': df['val_f1'].iloc[-1],
        'Test F1-score': df['test_f1'].iloc[-1]
    }
    
    x = list(final_metrics.keys())
    y = list(final_metrics.values())
    colors = ['blue', 'red', 'green', 'blue', 'red', 'green']
    
    bars = axes[0, 0].bar(x, y, color=colors, alpha=0.7)
    axes[0, 0].set_title('Final Performance Comparison')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add values on bars
    for bar, value in zip(bars, y):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                       f'{value:.3f}', ha='center', va='bottom')
    
    # 2. Best performance comparison
    best_metrics = {
        'Best Val Accuracy': df['val_acc'].max(),
        'Best Test Accuracy': df['test_acc'].max(),
        'Best Val F1-score': df['val_f1'].max(),
        'Best Test F1-score': df['test_f1'].max()
    }
    
    x = list(best_metrics.keys())
    y = list(best_metrics.values())
    colors = ['red', 'green', 'red', 'green']
    
    bars = axes[0, 1].bar(x, y, color=colors, alpha=0.7)
    axes[0, 1].set_title('Best Performance Comparison')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Add values on bars
    for bar, value in zip(bars, y):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                       f'{value:.3f}', ha='center', va='bottom')
    
    # 3. Convergence analysis
    # Calculate convergence speed (epoch to reach 90% of best performance)
    best_val_acc = df['val_acc'].max()
    convergence_epoch = df[df['val_acc'] >= 0.9 * best_val_acc]['epoch'].iloc[0] if len(df[df['val_acc'] >= 0.9 * best_val_acc]) > 0 else df['epoch'].iloc[-1]
    
    axes[1, 0].plot(df['epoch'], df['val_acc'], label='Validation Accuracy', color='red')
    axes[1, 0].axhline(y=0.9 * best_val_acc, color='black', linestyle='--', alpha=0.5, label='90% Best Performance')
    axes[1, 0].axvline(x=convergence_epoch, color='green', linestyle='--', alpha=0.7, label=f'Convergence Epoch: {convergence_epoch}')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Validation Accuracy')
    axes[1, 0].set_title('Convergence Analysis')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Overfitting analysis
    overfitting_gap = df['train_acc'] - df['val_acc']
    axes[1, 1].plot(df['epoch'], overfitting_gap, label='Overfitting Gap', color='orange')
    axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Train - Val Accuracy')
    axes[1, 1].set_title('Overfitting Analysis')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save image
    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, 'performance_summary.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Performance summary saved to: {save_path}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Training Accuracy and F1-score Visualization')
    parser.add_argument('--csv_file', type=str, required=True, help='CSV file path')
    parser.add_argument('--save_dir', type=str, default=None, help='Directory to save images')
    parser.add_argument('--plot_type', type=str, default='all', 
                       choices=['curves', 'statistics', 'summary', 'all'], help='Plot type')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.csv_file):
        print(f"Error: CSV file does not exist: {args.csv_file}")
        return
    
    print(f"Processing CSV file: {args.csv_file}")
    
    if args.plot_type in ['curves', 'all']:
        print("Plotting accuracy curves...")
        plot_accuracy_curves(args.csv_file, args.save_dir)
    
    if args.plot_type in ['statistics', 'all']:
        print("Plotting accuracy statistics...")
        plot_accuracy_statistics(args.csv_file, args.save_dir)
    
    if args.plot_type in ['summary', 'all']:
        print("Plotting performance summary...")
        plot_performance_summary(args.csv_file, args.save_dir)

if __name__ == "__main__":
    main() 