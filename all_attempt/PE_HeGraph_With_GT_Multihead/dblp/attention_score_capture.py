#!/usr/bin/env python3
"""
注意力分数捕获工具 - 专门适配MetapathMultiHeadAttentionLayer架构
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.utils import scatter
import warnings
warnings.filterwarnings('ignore')
import os

class AttentionScoreCapture:
    """注意力分数捕获器"""
    
    def __init__(self, model):
        """
        初始化捕获器
        
        Args:
            model: 训练好的模型
        """
        self.model = model
        self.attention_scores = {}
        self.hooks = []
        
    def register_hooks(self):
        """注册钩子来捕获注意力分数"""
        print("注册注意力分数捕获钩子...")
        
        # 清除之前的钩子
        self._remove_hooks()
        
        # 为每一层的注意力层注册钩子
        if hasattr(self.model, 'layers') and self.model.layers is not None:
            for layer_idx, layer in enumerate(self.model.layers):
                if hasattr(layer, 'attention'):
                    # 注册前向钩子
                    hook = layer.attention.register_forward_hook(
                        self._attention_forward_hook(layer_idx)
                    )
                    self.hooks.append(hook)
                    print(f"已为层 {layer_idx+1} 注册注意力钩子")
    
    def _attention_forward_hook(self, layer_idx):
        """注意力前向钩子"""
        def hook(module, input, output):
            # 捕获注意力分数
            self.attention_scores[f'layer_{layer_idx+1}'] = {
                'input': input,
                'output': output,
                'head_weights': module.head_weights.detach().cpu().numpy() if hasattr(module, 'head_weights') else None
            }
        return hook
    
    def _remove_hooks(self):
        """移除所有钩子"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def capture_attention_scores(self, data, pe_features=None):
        """
        捕获注意力分数
        
        Args:
            data: 输入数据
            pe_features: PE特征列表
            
        Returns:
            注意力分数字典
        """
        print("开始捕获注意力分数...")
        
        # 注册钩子
        self.register_hooks()
        
        # 前向传播
        with torch.no_grad():
            if hasattr(self.model, 'forward'):
                output = self.model.forward(data)
        
        # 移除钩子
        self._remove_hooks()
        
        print(f"捕获完成，共捕获 {len(self.attention_scores)} 层的注意力分数")
        return self.attention_scores

class MetapathAttentionAnalyzer:
    """基于元路径的注意力分析器"""
    
    def __init__(self, model, device='cuda'):
        """
        初始化分析器
        
        Args:
            model: 训练好的模型
            device: 计算设备
        """
        self.model = model
        self.device = device
        self.capture = AttentionScoreCapture(model)
        
    def analyze_attention_patterns(self, data, save_dir=None):
        """
        分析注意力模式
        
        Args:
            data: 输入数据
            save_dir: 保存目录
            
        Returns:
            分析结果
        """
        print("\n=== 注意力模式分析 ===")
        
        # 捕获注意力分数
        attention_scores = self.capture.capture_attention_scores(data)
        
        results = {}
        
        for layer_name, layer_data in attention_scores.items():
            print(f"\n分析 {layer_name}:")
            
            # 分析头权重
            if layer_data['head_weights'] is not None:
                weights = layer_data['head_weights'].flatten()
                print(f"  头权重: {weights}")
                print(f"  权重均值: {np.mean(weights):.4f}")
                print(f"  权重标准差: {np.std(weights):.4f}")
                
                results[layer_name] = {
                    'head_weights': weights.tolist(),
                    'weight_mean': float(np.mean(weights)),
                    'weight_std': float(np.std(weights)),
                    'weight_ranking': np.argsort(weights)[::-1].tolist()
                }
        
        # 可视化分析结果
        if save_dir:
            self._visualize_attention_analysis(results, save_dir)
        
        return results
    
    def _visualize_attention_analysis(self, results, save_dir):
        """可视化注意力分析结果"""
        if not results:
            return
        
        # 1. 头权重热力图
        self._plot_head_weights_heatmap(results, save_dir)
        
        # 2. 头权重演化图
        self._plot_head_weights_evolution(results, save_dir)
        
        # 3. 头重要性排名
        self._plot_head_importance_ranking(results, save_dir)
    
    def _plot_head_weights_heatmap(self, results, save_dir):
        """绘制头权重热力图"""
        # 准备数据
        layers = list(results.keys())
        head_names = ['Node_Feature', 'PE_1', 'PE_2', 'PE_3', 'PE_4']  # 根据实际头数调整
        
        # 创建权重矩阵
        weights_matrix = np.array([results[layer]['head_weights'] for layer in layers])
        
        # 绘制热力图
        plt.figure(figsize=(12, 8))
        sns.heatmap(weights_matrix, 
                   xticklabels=head_names[:weights_matrix.shape[1]],
                   yticklabels=layers,
                   annot=True, 
                   fmt='.3f',
                   cmap='YlOrRd',
                   cbar_kws={'label': 'Head Weight'})
        
        plt.title('头权重热力图')
        plt.xlabel('头类型')
        plt.ylabel('层')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        # 保存图片
        save_path = os.path.join(save_dir, 'attention_head_weights_heatmap.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"头权重热力图已保存: {save_path}")
    
    def _plot_head_weights_evolution(self, results, save_dir):
        """绘制头权重演化图"""
        # 准备数据
        layers = list(results.keys())
        head_names = ['Node_Feature', 'PE_1', 'PE_2', 'PE_3', 'PE_4']  # 根据实际头数调整
        
        # 创建权重矩阵
        weights_matrix = np.array([results[layer]['head_weights'] for layer in layers])
        
        # 绘制演化图
        plt.figure(figsize=(12, 6))
        for i, head_name in enumerate(head_names[:weights_matrix.shape[1]]):
            plt.plot(range(1, len(layers)+1), weights_matrix[:, i], 
                    marker='o', linewidth=2, label=head_name)
        
        plt.title('头权重演化图')
        plt.xlabel('层')
        plt.ylabel('权重')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 保存图片
        save_path = os.path.join(save_dir, 'attention_head_weights_evolution.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"头权重演化图已保存: {save_path}")
    
    def _plot_head_importance_ranking(self, results, save_dir):
        """绘制头重要性排名"""
        # 计算平均重要性
        head_names = ['Node_Feature', 'PE_1', 'PE_2', 'PE_3', 'PE_4']  # 根据实际头数调整
        avg_weights = np.zeros(len(head_names))
        
        for layer_data in results.values():
            weights = np.array(layer_data['head_weights'])
            avg_weights[:len(weights)] += weights
        avg_weights /= len(results)
        
        # 排序
        sorted_indices = np.argsort(avg_weights)[::-1]
        sorted_weights = avg_weights[sorted_indices]
        sorted_names = [head_names[i] for i in sorted_indices]
        
        # 绘制排名图
        plt.figure(figsize=(10, 6))
        bars = plt.bar(range(len(sorted_names)), sorted_weights, 
                      color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'][:len(sorted_names)])
        
        plt.title('头重要性排名')
        plt.xlabel('头类型')
        plt.ylabel('平均权重')
        plt.xticks(range(len(sorted_names)), sorted_names, rotation=45)
        
        # 添加数值标签
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # 保存图片
        save_path = os.path.join(save_dir, 'attention_head_importance_ranking.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"头重要性排名图已保存: {save_path}")

def create_attention_visualization_tool(model, data, save_dir):
    """
    创建注意力可视化工具
    
    Args:
        model: 训练好的模型
        data: 输入数据
        save_dir: 保存目录
    """
    print("创建注意力可视化工具...")
    
    # 创建分析器
    analyzer = MetapathAttentionAnalyzer(model)
    
    # 分析注意力模式
    results = analyzer.analyze_attention_patterns(data, save_dir)
    
    print("注意力可视化工具创建完成")
    return results

# 传统attention score分析方法的实现
class TraditionalAttentionAnalyzer:
    """传统注意力分数分析器"""
    
    def __init__(self, model):
        self.model = model
        self.attention_scores = {}
    
    def capture_traditional_attention_scores(self, data):
        """
        捕获传统的注意力分数（Q*K^T）
        
        Args:
            data: 输入数据
            
        Returns:
            注意力分数字典
        """
        print("捕获传统注意力分数...")
        
        # 这里需要根据具体的模型架构来实现
        # 传统方法通常计算 Q * K^T 的分数
        
        attention_scores = {}
        
        # 示例实现（需要根据具体模型调整）
        if hasattr(self.model, 'layers') and self.model.layers is not None:
            for layer_idx, layer in enumerate(self.model.layers):
                if hasattr(layer, 'attention'):
                    # 获取Q, K, V
                    # 这里需要根据您的具体实现来获取
                    print(f"层 {layer_idx+1}: 需要根据具体模型架构来获取Q, K, V")
        
        return attention_scores
    
    def visualize_attention_scores(self, attention_scores, save_dir):
        """
        可视化注意力分数
        
        Args:
            attention_scores: 注意力分数字典
            save_dir: 保存目录
        """
        print("可视化注意力分数...")
        
        for layer_name, scores in attention_scores.items():
            if isinstance(scores, np.ndarray):
                # 绘制注意力分数热力图
                plt.figure(figsize=(10, 8))
                sns.heatmap(scores, 
                           cmap='YlOrRd',
                           cbar_kws={'label': 'Attention Score'})
                plt.title(f'{layer_name} 注意力分数热力图')
                plt.xlabel('Key')
                plt.ylabel('Query')
                
                # 保存图片
                save_path = os.path.join(save_dir, f'{layer_name}_attention_heatmap.png')
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"注意力分数热力图已保存: {save_path}")

if __name__ == "__main__":
    print("注意力分数捕获工具")
    print("="*50)
    print("此工具专门用于分析基于元路径的多头注意力机制")
    print("请根据您的具体模型和数据来使用此工具") 