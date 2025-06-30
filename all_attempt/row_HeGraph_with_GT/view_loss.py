"""
Training Loss Visualization Tool
For plotting loss curves during training process
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
from pathlib import Path

def plot_loss_curves(csv_file, save_dir=None):
    """
    Plot loss curves
    
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
    
    # Plot loss curves
    ax1.plot(df['epoch'], df['train_loss'], label='Training Loss', linewidth=2, color='blue')
    ax1.plot(df['epoch'], df['val_loss'], label='Validation Loss', linewidth=2, color='red')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss Curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot loss difference
    loss_diff = df['train_loss'] - df['val_loss']
    ax2.plot(df['epoch'], loss_diff, label='Train Loss - Val Loss', linewidth=2, color='green')
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss Difference')
    ax2.set_title('Training and Validation Loss Difference')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save image
    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, 'loss_curves.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Loss curves saved to: {save_path}")
    
    plt.show()

def plot_loss_statistics(csv_file, save_dir=None):
    """
    Plot loss statistics
    
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
    
    # 1. Loss distribution histogram
    axes[0, 0].hist(df['train_loss'], bins=20, alpha=0.7, label='Training Loss', color='blue')
    axes[0, 0].hist(df['val_loss'], bins=20, alpha=0.7, label='Validation Loss', color='red')
    axes[0, 0].set_xlabel('Loss')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Loss Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Loss boxplot
    loss_data = [df['train_loss'], df['val_loss']]
    axes[0, 1].boxplot(loss_data, labels=['Training Loss', 'Validation Loss'])
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Loss Boxplot')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Loss change rate
    train_loss_change = df['train_loss'].diff()
    val_loss_change = df['val_loss'].diff()
    axes[1, 0].plot(df['epoch'][1:], train_loss_change[1:], label='Training Loss Change', color='blue')
    axes[1, 0].plot(df['epoch'][1:], val_loss_change[1:], label='Validation Loss Change', color='red')
    axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss Change')
    axes[1, 0].set_title('Loss Change Rate')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Loss correlation scatter plot
    axes[1, 1].scatter(df['train_loss'], df['val_loss'], alpha=0.6, color='purple')
    axes[1, 1].set_xlabel('Training Loss')
    axes[1, 1].set_ylabel('Validation Loss')
    axes[1, 1].set_title('Training Loss vs Validation Loss')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add correlation coefficient
    corr = df['train_loss'].corr(df['val_loss'])
    axes[1, 1].text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=axes[1, 1].transAxes, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    plt.tight_layout()
    
    # Save image
    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, 'loss_statistics.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Loss statistics saved to: {save_path}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Training Loss Visualization')
    parser.add_argument('--csv_file', type=str, required=True, help='CSV file path')
    parser.add_argument('--save_dir', type=str, default=None, help='Directory to save images')
    parser.add_argument('--plot_type', type=str, default='both', 
                       choices=['curves', 'statistics', 'both'], help='Plot type')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.csv_file):
        print(f"Error: CSV file does not exist: {args.csv_file}")
        return
    
    print(f"Processing CSV file: {args.csv_file}")
    
    if args.plot_type in ['curves', 'both']:
        print("Plotting loss curves...")
        plot_loss_curves(args.csv_file, args.save_dir)
    
    if args.plot_type in ['statistics', 'both']:
        print("Plotting loss statistics...")
        plot_loss_statistics(args.csv_file, args.save_dir)

if __name__ == "__main__":
    main() 