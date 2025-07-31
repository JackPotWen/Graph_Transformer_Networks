#!/usr/bin/env python3
"""
Attention Score Analysis Tool
Analyzes attention scores from QKV computation in MetapathMultiHeadAttentionLayer
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import pandas as pd
from scipy.special import softmax

class AttentionScoreAnalyzer:
    """Analyzes attention scores from QKV computation"""
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.attention_scores = {}
        self.head_names = []
        self.num_layers = 0
        
        # Ensure model is on the correct device
        self.model = self.model.to(self.device)
        
        self._extract_model_info()
    
    def _extract_model_info(self):
        """Extract model information"""
        self.num_layers = len(self.model.layers) if hasattr(self.model, 'layers') else 0
        
        # Extract head names based on metapath information
        if hasattr(self.model, 'layers') and self.model.layers is not None:
            for layer in self.model.layers:
                if hasattr(layer, 'attention') and hasattr(layer.attention, 'num_heads'):
                    num_heads = layer.attention.num_heads
                    self.head_names = ['Node_Feature'] + [f'PE_{j+1}' for j in range(num_heads-1)]
                    break
        
        print(f"Model information extracted:")
        print(f"  - Number of layers: {self.num_layers}")
        print(f"  - Head names: {self.head_names}")
    
    def capture_attention_scores(self, data):
        """Capture attention scores from all layers"""
        print("\n=== Capturing Attention Scores ===")
        
        # Ensure data is on the same device as model
        data = data.to(self.device)
        
        # Enable attention capture for all layers
        for layer in self.model.layers:
            if hasattr(layer, 'attention'):
                layer.attention.enable_attention_capture()
        
        # Forward pass to capture attention scores
        with torch.no_grad():
            _ = self.model(data)
        
        # Collect attention scores from all layers
        for layer_idx, layer in enumerate(self.model.layers):
            if hasattr(layer, 'attention'):
                scores = layer.attention.get_attention_scores()
                if scores:
                    self.attention_scores[f'layer_{layer_idx+1}'] = scores
                    print(f"Layer {layer_idx+1}: Captured {len(scores)} attention heads")
        
        # Disable attention capture
        for layer in self.model.layers:
            if hasattr(layer, 'attention'):
                layer.attention.disable_attention_capture()
        
        return self.attention_scores
    
    def analyze_attention_patterns(self, save_dir: str = None) -> Dict:
        """Analyze attention score patterns"""
        print("\n=== Attention Pattern Analysis ===")
        
        if not self.attention_scores:
            print("Warning: No attention scores captured")
            return {}
        
        results = {}
        
        for layer_name, layer_scores in self.attention_scores.items():
            print(f"\n{layer_name}:")
            layer_results = {}
            
            for head_idx, scores in enumerate(layer_scores):
                head_name = self.head_names[head_idx] if head_idx < len(self.head_names) else f'head_{head_idx}'
                
                # Convert to numpy for analysis
                scores_np = scores.numpy()
                
                # Basic statistics
                mean_score = np.mean(scores_np)
                std_score = np.std(scores_np)
                min_score = np.min(scores_np)
                max_score = np.max(scores_np)
                
                print(f"  {head_name}:")
                print(f"    Mean: {mean_score:.4f}")
                print(f"    Std: {std_score:.4f}")
                print(f"    Min: {min_score:.4f}")
                print(f"    Max: {max_score:.4f}")
                
                layer_results[head_name] = {
                    'scores': scores_np.tolist(),
                    'mean': float(mean_score),
                    'std': float(std_score),
                    'min': float(min_score),
                    'max': float(max_score),
                    'attention_entropy': self._calculate_attention_entropy(scores_np)
                }
            
            results[layer_name] = layer_results
        
        # Visualize attention patterns
        if save_dir:
            self._plot_attention_heatmap(results, save_dir)
            self._plot_attention_distribution(results, save_dir)
            self._plot_attention_entropy(results, save_dir)
        
        return results
    
    def _calculate_attention_entropy(self, scores: np.ndarray) -> float:
        """Calculate attention entropy as a measure of attention diversity"""
        # Normalize scores to probabilities
        scores_normalized = scores / (np.sum(scores) + 1e-8)
        # Calculate entropy
        entropy = -np.sum(scores_normalized * np.log(scores_normalized + 1e-8))
        return float(entropy)
    
    def _plot_attention_heatmap(self, results: Dict, save_dir: str):
        """Plot attention score heatmap"""
        if not results:
            print("Warning: Cannot plot attention heatmap - no results data")
            return
        
        # Prepare data for heatmap
        layers = list(results.keys())
        head_names = list(results[layers[0]].keys()) if layers else []
        
        if not head_names:
            print("Warning: Cannot plot attention heatmap - no head data")
            return
        
        # Create mean attention score matrix
        attention_matrix = np.zeros((len(layers), len(head_names)))
        for i, layer in enumerate(layers):
            for j, head in enumerate(head_names):
                if head in results[layer]:
                    attention_matrix[i, j] = results[layer][head]['mean']
        
        # Create original heatmap (all heads)
        plt.figure(figsize=(12, 8))
        sns.heatmap(attention_matrix, 
                   xticklabels=head_names,
                   yticklabels=layers,
                   annot=True, 
                   fmt='.3f',
                   cmap='YlOrRd',
                   cbar_kws={'label': 'Mean Attention Score'})
        
        plt.title('Attention Score Heatmap (All Heads)')
        plt.xlabel('Head Type')
        plt.ylabel('Layer')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        # Save plot
        save_path = os.path.join(save_dir, 'attention_score_heatmap.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Attention score heatmap saved: {save_path}")
        
        # Create PE-only heatmap with softmax normalization
        pe_head_names = [head for head in head_names if head.startswith('PE_')]
        if len(pe_head_names) >= 2:  # Need at least 2 PE heads for meaningful comparison
            pe_attention_matrix = np.zeros((len(layers), len(pe_head_names)))
            
            for i, layer in enumerate(layers):
                for j, head in enumerate(pe_head_names):
                    if head in results[layer]:
                        pe_attention_matrix[i, j] = results[layer][head]['mean']
            
            # Apply softmax normalization across PE heads for each layer
            pe_attention_matrix_softmax = np.zeros_like(pe_attention_matrix)
            for i in range(pe_attention_matrix.shape[0]):
                if np.sum(pe_attention_matrix[i, :]) > 0:  # Avoid division by zero
                    pe_attention_matrix_softmax[i, :] = softmax(pe_attention_matrix[i, :])
                else:
                    pe_attention_matrix_softmax[i, :] = pe_attention_matrix[i, :]
            
            # Create PE-only heatmap with softmax
            plt.figure(figsize=(10, 6))
            sns.heatmap(pe_attention_matrix_softmax, 
                       xticklabels=pe_head_names,
                       yticklabels=layers,
                       annot=True, 
                       fmt='.3f',
                       cmap='YlOrRd',
                       cbar_kws={'label': 'Softmax Normalized PE Attention Score'})
            
            plt.title('PE Attention Score Heatmap (Softmax Normalized)')
            plt.xlabel('PE Head Type')
            plt.ylabel('Layer')
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
            
            # Save PE-only plot
            save_path_pe = os.path.join(save_dir, 'pe_attention_heatmap_softmax.png')
            plt.savefig(save_path_pe, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"PE attention heatmap (softmax) saved: {save_path_pe}")
            
            # Also create a version with original values for comparison
            plt.figure(figsize=(10, 6))
            sns.heatmap(pe_attention_matrix, 
                       xticklabels=pe_head_names,
                       yticklabels=layers,
                       annot=True, 
                       fmt='.3f',
                       cmap='YlOrRd',
                       cbar_kws={'label': 'Original PE Attention Score'})
            
            plt.title('PE Attention Score Heatmap (Original Values)')
            plt.xlabel('PE Head Type')
            plt.ylabel('Layer')
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
            
            # Save original PE plot
            save_path_pe_orig = os.path.join(save_dir, 'pe_attention_heatmap_original.png')
            plt.savefig(save_path_pe_orig, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"PE attention heatmap (original) saved: {save_path_pe_orig}")
        else:
            print("Warning: Not enough PE heads found for PE-only heatmap")
        
        # Create cross-layer PE importance heatmap (1 x k)
        if len(pe_head_names) >= 2:
            # Calculate weighted average across layers for each PE
            pe_cross_layer_importance = np.zeros(len(pe_head_names))
            
            # Weight by layer depth (deeper layers get higher weight)
            layer_weights = np.linspace(1.0, 2.0, len(layers))  # 1.0 to 2.0 for layer weights
            
            for pe_idx, pe_head in enumerate(pe_head_names):
                pe_values = []
                pe_weights = []
                
                for layer_idx, layer in enumerate(layers):
                    if pe_head in results[layer]:
                        pe_values.append(results[layer][pe_head]['mean'])
                        pe_weights.append(layer_weights[layer_idx])
                
                if pe_values:
                    # Calculate weighted average
                    pe_cross_layer_importance[pe_idx] = np.average(pe_values, weights=pe_weights)
            
            # Apply softmax to get relative importance
            pe_importance_softmax = softmax(pe_cross_layer_importance)
            
            # Create 1 x k heatmap
            plt.figure(figsize=(10, 3))
            heatmap_data = pe_importance_softmax.reshape(1, -1)
            
            sns.heatmap(heatmap_data, 
                       xticklabels=pe_head_names,
                       yticklabels=['Cross-Layer PE Importance'],
                       annot=True, 
                       fmt='.3f',
                       cmap='YlOrRd',
                       cbar_kws={'label': 'Softmax Normalized Cross-Layer PE Importance'})
            
            plt.title('Cross-Layer PE Importance Heatmap (1 × k)')
            plt.xlabel('PE Head Type')
            plt.ylabel('')
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
            
            # Save cross-layer PE importance plot
            save_path_cross = os.path.join(save_dir, 'pe_cross_layer_importance_heatmap.png')
            plt.savefig(save_path_cross, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Cross-layer PE importance heatmap saved: {save_path_cross}")
            
            # Also create a version with original weighted averages for comparison
            plt.figure(figsize=(10, 3))
            heatmap_data_orig = pe_cross_layer_importance.reshape(1, -1)
            
            sns.heatmap(heatmap_data_orig, 
                       xticklabels=pe_head_names,
                       yticklabels=['Cross-Layer PE Importance'],
                       annot=True, 
                       fmt='.3f',
                       cmap='YlOrRd',
                       cbar_kws={'label': 'Original Weighted Average PE Importance'})
            
            plt.title('Cross-Layer PE Importance Heatmap (Original Values)')
            plt.xlabel('PE Head Type')
            plt.ylabel('')
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
            
            # Save original cross-layer plot
            save_path_cross_orig = os.path.join(save_dir, 'pe_cross_layer_importance_original.png')
            plt.savefig(save_path_cross_orig, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Cross-layer PE importance heatmap (original) saved: {save_path_cross_orig}")
            
            # Save detailed analysis data
            cross_layer_analysis = {
                'pe_head_names': pe_head_names,
                'weighted_averages': pe_cross_layer_importance.tolist(),
                'softmax_normalized': pe_importance_softmax.tolist(),
                'layer_weights': layer_weights.tolist(),
                'analysis': {
                    'most_important_pe': pe_head_names[np.argmax(pe_importance_softmax)],
                    'least_important_pe': pe_head_names[np.argmin(pe_importance_softmax)],
                    'importance_ratio': float(np.max(pe_importance_softmax) / np.min(pe_importance_softmax)),
                    'importance_std': float(np.std(pe_importance_softmax))
                }
            }
            
            # Save to JSON
            analysis_path = os.path.join(save_dir, 'pe_cross_layer_analysis.json')
            with open(analysis_path, 'w') as f:
                json.dump(cross_layer_analysis, f, indent=2)
            print(f"Cross-layer PE analysis data saved: {analysis_path}")
            
            # Print summary
            print(f"\n=== Cross-Layer PE Importance Summary ===")
            print(f"Most important PE: {cross_layer_analysis['analysis']['most_important_pe']}")
            print(f"Least important PE: {cross_layer_analysis['analysis']['least_important_pe']}")
            print(f"Importance ratio: {cross_layer_analysis['analysis']['importance_ratio']:.3f}")
            print(f"Importance std: {cross_layer_analysis['analysis']['importance_std']:.3f}")
            print(f"Softmax values: {[f'{v:.3f}' for v in pe_importance_softmax]}")
        else:
            print("Warning: Not enough PE heads found for cross-layer analysis")
    
    def _plot_attention_distribution(self, results: Dict, save_dir: str):
        """Plot attention score distribution"""
        if not results:
            print("Warning: Cannot plot attention distribution - no results data")
            return
        
        # Create subplots for each layer
        num_layers = len(results)
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, (layer_name, layer_data) in enumerate(results.items()):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            head_names = list(layer_data.keys())
            
            # Plot distribution for each head
            for head_name in head_names:
                scores = layer_data[head_name]['scores']
                ax.hist(scores, bins=50, alpha=0.7, label=head_name, density=True)
            
            ax.set_title(f'{layer_name} Attention Distribution')
            ax.set_xlabel('Attention Score')
            ax.set_ylabel('Density')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(num_layers, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        # Save plot
        save_path = os.path.join(save_dir, 'attention_score_distribution.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Attention score distribution saved: {save_path}")
    
    def _plot_attention_entropy(self, results: Dict, save_dir: str):
        """Plot attention entropy analysis"""
        if not results:
            print("Warning: Cannot plot attention entropy - no results data")
            return
        
        # Prepare data
        layers = list(results.keys())
        head_names = list(results[layers[0]].keys()) if layers else []
        
        if not head_names:
            print("Warning: Cannot plot attention entropy - no head data")
            return
        
        # Create entropy matrix
        entropy_matrix = np.zeros((len(layers), len(head_names)))
        for i, layer in enumerate(layers):
            for j, head in enumerate(head_names):
                if head in results[layer]:
                    entropy_matrix[i, j] = results[layer][head]['attention_entropy']
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(entropy_matrix, 
                   xticklabels=head_names,
                   yticklabels=layers,
                   annot=True, 
                   fmt='.3f',
                   cmap='Blues',
                   cbar_kws={'label': 'Attention Entropy'})
        
        plt.title('Attention Entropy Heatmap')
        plt.xlabel('Head Type')
        plt.ylabel('Layer')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        # Save plot
        save_path = os.path.join(save_dir, 'attention_entropy_heatmap.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Attention entropy heatmap saved: {save_path}")
    
    def comprehensive_analysis(self, data, save_dir: str = None) -> Dict:
        """Run comprehensive attention analysis"""
        print("\n" + "="*60)
        print("Starting Comprehensive Attention Analysis")
        print("="*60)
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        # Capture attention scores
        self.capture_attention_scores(data)
        
        # Analyze patterns
        analysis_results = self.analyze_attention_patterns(save_dir)
        
        # Generate report
        if save_dir:
            self._generate_analysis_report(analysis_results, save_dir)
        
        return analysis_results
    
    def _generate_analysis_report(self, results: Dict, save_dir: str):
        """Generate comprehensive analysis report"""
        report = {
            'analysis_timestamp': datetime.now().isoformat(),
            'model_info': {
                'num_layers': self.num_layers,
                'head_names': self.head_names
            },
            'results': results
        }
        
        # Save JSON report
        report_path = os.path.join(save_dir, 'attention_analysis_report.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Generate text report
        text_report_path = os.path.join(save_dir, 'attention_analysis_report.txt')
        with open(text_report_path, 'w', encoding='utf-8') as f:
            f.write("Attention Score Analysis Report\n")
            f.write("="*50 + "\n\n")
            
            f.write(f"Analysis Time: {report['analysis_timestamp']}\n")
            f.write(f"Number of Layers: {report['model_info']['num_layers']}\n")
            f.write(f"Head Names: {report['model_info']['head_names']}\n\n")
            
            # Attention analysis results
            if results:
                f.write("Attention Analysis Results:\n")
                f.write("-"*30 + "\n")
                for layer_name, layer_data in results.items():
                    f.write(f"{layer_name}:\n")
                    for head_name, head_data in layer_data.items():
                        f.write(f"  {head_name}:\n")
                        f.write(f"    Mean: {head_data['mean']:.4f}\n")
                        f.write(f"    Std: {head_data['std']:.4f}\n")
                        f.write(f"    Min: {head_data['min']:.4f}\n")
                        f.write(f"    Max: {head_data['max']:.4f}\n")
                        f.write(f"    Entropy: {head_data['attention_entropy']:.4f}\n")
                    f.write("\n")
            else:
                f.write("Analysis Results:\n")
                f.write("-"*30 + "\n")
                f.write("No attention scores captured.\n")
                f.write("Please ensure the model has attention layers.\n\n")
        
        print(f"Comprehensive analysis report saved:")
        print(f"  JSON report: {report_path}")
        print(f"  Text report: {text_report_path}")


def analyze_model_attention(model_path: str, data_path: str, save_dir: str = None):
    """Standalone function to analyze model attention scores"""
    # Load model and data
    # This is a placeholder - implement based on your specific model loading logic
    pass


if __name__ == "__main__":
    print("Attention Score Analysis Tool")
    print("This module provides tools for analyzing attention scores from QKV computation") 