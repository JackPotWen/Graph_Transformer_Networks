#!/usr/bin/env python3
"""
基于元路径的多头注意力异构图Transformer训练脚本
头数与PE数量绑定：头数 = 1 + k (k为PE数量)
"""

import torch
import torch.nn as nn
import numpy as np
import os
import time
import json
import argparse
from datetime import datetime

# 导入自定义模块
from load_data import prepare_heterogeneous_data
from HGT import MetapathGraphTransformerNet
from train_utils import train_model, save_results, view_model_param

# 导入头贡献分析工具
try:
    from attention_score_analysis import AttentionScoreAnalyzer
    ATTENTION_ANALYSIS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Attention analysis tool import failed: {e}")
    print("Will skip attention analysis functionality")
    ATTENTION_ANALYSIS_AVAILABLE = False

def gpu_setup(use_gpu, gpu_id):
    """GPU设置"""
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  

    if torch.cuda.is_available() and use_gpu:
        print('CUDA可用，GPU:', torch.cuda.get_device_name(0))
        device = torch.device("cuda")
    else:
        print('CUDA不可用，使用CPU')
        device = torch.device("cpu")
    return device

def get_default_params():
    """获取默认训练参数"""
    params = {
        'seed': 42,
        'epochs': 50,
        'init_lr': 0.001,
        'lr_reduce_factor': 0.5,
        'lr_schedule_patience': 10,
        'min_lr': 1e-6,
        'weight_decay': 1e-4,
        'max_time': 12,  # 最大训练时间（小时）
    }
    return params

def get_default_net_params(device, in_dim, n_classes):
    """获取默认网络参数"""
    net_params = {
        'L': 4,  # 层数
        'hidden_dim': 128,  # 隐藏层维度
        'out_dim': 128,  # 输出维度
        'in_feat_dropout': 0.1,  # 输入特征dropout
        'dropout': 0.1,  # dropout
        'layer_norm': False,  # 是否使用层归一化
        'batch_norm': True,  # 是否使用批归一化
        'residual': True,  # 是否使用残差连接
        'edge_feat_dim': 1,  # 边特征维度
        'readout': 'mean',  # 读出方式
        'lap_pos_enc': False,  # 是否使用拉普拉斯位置编码
        'wl_pos_enc': False,  # 是否使用WL位置编码
        'metapath2vec_pos_enc': True,  # 是否使用metapath2vec位置编码
        'pos_enc_dim': 8,  # 位置编码维度
        # 支持多个metapath2vec PE路径
        'metapath2vec_pe_paths': [
            '/home/kuei-jan/github/Graph_Transformer_Networks/metapath2vec_including_edge_weight/weighted_metapath2vec/experiment_result_with_weighted_sample/vec_feature_apa_w200_l50_d128_c5_ns5.pkl',
            '/home/kuei-jan/github/Graph_Transformer_Networks/metapath2vec_including_edge_weight/weighted_metapath2vec/experiment_result_with_weighted_sample/vec_feature_pap_w200_l50_d128_c5_ns5.pkl',
            '/home/kuei-jan/github/Graph_Transformer_Networks/metapath2vec_including_edge_weight/weighted_metapath2vec/experiment_result_with_weighted_sample/vec_feature_apapa_w200_l50_d128_c5_ns5.pkl',
            '/home/kuei-jan/github/Graph_Transformer_Networks/metapath2vec_including_edge_weight/weighted_metapath2vec/experiment_result_with_weighted_sample/vec_feature_apcpa_w200_l50_d128_c5_ns5.pkl'
        ],
        'metapath_names': ['APA', 'PAP', 'APAPA', 'APCPA'], # 对应的元路径名称
        'node_feat_is_int': False,  # 节点特征是否为整数
        'device': device,
        'in_dim': in_dim,
        'n_classes': n_classes,
        # PE融合模式控制
        'pe_fusion_mode': 'concat',  # 可选: 'add' 或 'concat'
    }
    return net_params

def run_attention_analysis(model, data, save_dir, device):
    """
    Run attention score analysis
    
    Args:
        model: Trained model
        data: Input data
        save_dir: Save directory
        device: Computing device
        
    Returns:
        Analysis results
    """
    print("\n" + "="*60)
    print("Starting Attention Score Analysis")
    print("="*60)
    
    if not ATTENTION_ANALYSIS_AVAILABLE:
        print("Error: Attention analysis tool not available")
        return None
    
    try:
        # Create analysis directory
        analysis_dir = os.path.join(save_dir, 'attention_analysis')
        os.makedirs(analysis_dir, exist_ok=True)
        
        print(f"Attention analysis results will be saved to: {analysis_dir}")
        
        # Attention score analyzer
        print("\n" + "="*40)
        print("1. Attention Score Analyzer")
        print("="*40)
        
        analyzer = AttentionScoreAnalyzer(model, device)
        
        # Run comprehensive analysis
        attention_results = analyzer.comprehensive_analysis(data, analysis_dir)
        
        print("\nAttention analysis completed!")
        
        print(f"\nAll attention analysis completed! Results saved to: {analysis_dir}")
        
        return {
            'attention_results': attention_results,
            'analysis_dir': analysis_dir
        }
        
    except Exception as e:
        print(f"Error during attention analysis: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_analysis_summary(head_results, attention_results, analysis_dir):
    """
    生成分析总结
    
    Args:
        head_results: 头贡献分析结果
        attention_results: 注意力分析结果
        analysis_dir: 分析目录
    """
    print("生成分析总结...")
    
    summary = {
        'analysis_timestamp': datetime.now().isoformat(),
        'head_contribution_summary': {},
        'attention_pattern_summary': {},
        'key_findings': []
    }
    
    # 头贡献分析总结
    if head_results and 'head_weights' in head_results:
        head_weights = head_results['head_weights']
        summary['head_contribution_summary'] = {
            'num_layers': len(head_weights),
            'layer_analysis': {}
        }
        
        for layer_name, layer_data in head_weights.items():
            weights = layer_data['weights']
            ranking = layer_data['ranking']
            
            summary['head_contribution_summary']['layer_analysis'][layer_name] = {
                'weights': weights,
                'ranking': ranking,
                'most_important_head': ranking[0] if ranking else None,
                'least_important_head': ranking[-1] if ranking else None
            }
    
    # 消融实验总结
    if head_results and 'ablation_study' in head_results:
        ablation = head_results['ablation_study']
        summary['ablation_summary'] = {
            'num_layers': len(ablation),
            'performance_impact': {}
        }
        
        for layer_name, layer_data in ablation.items():
            max_drop = 0
            most_important_head = None
            
            for head_name, head_data in layer_data.items():
                if head_data['performance_drop'] > max_drop:
                    max_drop = head_data['performance_drop']
                    most_important_head = head_name
            
            summary['ablation_summary']['performance_impact'][layer_name] = {
                'most_important_head': most_important_head,
                'max_performance_drop': max_drop
            }
    
    # 注意力模式总结
    if attention_results:
        summary['attention_pattern_summary'] = {
            'num_layers': len(attention_results),
            'weight_statistics': {}
        }
        
        for layer_name, layer_data in attention_results.items():
            weights = layer_data['head_weights']
            summary['attention_pattern_summary']['weight_statistics'][layer_name] = {
                'mean_weight': layer_data['weight_mean'],
                'std_weight': layer_data['weight_std'],
                'weight_ranking': layer_data['weight_ranking']
            }
    
    # 关键发现
    key_findings = []
    
    # 分析头权重分布
    if head_results and 'head_weights' in head_results and head_results['head_weights']:
        all_weights = []
        for layer_data in head_results['head_weights'].values():
            all_weights.extend(layer_data['weights'])
        
        if all_weights:
            weight_variance = np.var(all_weights)
            if weight_variance > 0.1:
                key_findings.append("头权重存在显著差异，表明不同头的重要性不同")
            else:
                key_findings.append("头权重相对均匀，表明各头贡献相对平衡")
    
    # 分析消融实验结果
    if head_results and 'ablation_study' in head_results and head_results['ablation_study']:
        max_performance_drops = []
        for layer_data in head_results['ablation_study'].values():
            for head_data in layer_data.values():
                max_performance_drops.append(head_data['performance_drop'])
        
        if max_performance_drops:
            max_drop = max(max_performance_drops)
            if max_drop > 0.05:
                key_findings.append(f"消融实验显示某些头的移除会导致显著性能下降（最大下降: {max_drop:.4f}）")
            else:
                key_findings.append("消融实验显示模型对单个头的移除相对鲁棒")
    
    # 如果没有关键发现，添加说明
    if not key_findings:
        key_findings.append("未找到可分析的头权重信息，请确保模型使用了可训练的头权重")
    
    summary['key_findings'] = key_findings
    
    # 保存总结
    summary_path = os.path.join(analysis_dir, 'analysis_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # 打印总结
    print("\n分析总结:")
    print("-" * 30)
    print(f"分析时间: {summary['analysis_timestamp']}")
    print(f"分析层数: {summary['head_contribution_summary'].get('num_layers', 0)}")
    
    if key_findings:
        print("\n关键发现:")
        for i, finding in enumerate(key_findings, 1):
            print(f"  {i}. {finding}")
    
    print(f"\n详细总结已保存到: {summary_path}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='基于元路径的多头注意力异构图Transformer训练')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
    parser.add_argument('--use_gpu', action='store_true', help='是否使用GPU')
    parser.add_argument('--data_path', type=str, default='/home/kuei-jan/github/Graph_Transformer_Networks/data/DBLP', help='数据路径')
    parser.add_argument('--out_dir', type=str, default='/home/kuei-jan/github/Graph_Transformer_Networks/all_attempt/PE_HeGraph_With_GT_Multihead/dblp/true_dblp_result', help='输出目录')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--hidden_dim', type=int, default=128, help='隐藏层维度')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--pe_fusion_mode', type=str, default='concat', choices=['add', 'concat'], help='PE fusion mode')
    parser.add_argument('--enable_attention_analysis', action='store_true', help='Enable attention score analysis')
    parser.add_argument('--attention_analysis_only', action='store_true', help='Run attention analysis only (skip training)')
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # GPU设置
    device = gpu_setup(args.use_gpu, args.gpu_id)
    
    print("=" * 60)
    print("基于元路径的多头注意力异构图Transformer训练")
    print("头数与PE数量绑定：头数 = 1 + k (k为PE数量)")
    print("=" * 60)
    
    # 加载和准备数据
    print("正在加载数据...")
    # 先创建临时的网络参数来获取PE设置
    temp_device = device
    temp_in_dim = 10  # 临时值，后面会更新
    temp_n_classes = 4  # 临时值，后面会更新
    temp_net_params = get_default_net_params(temp_device, temp_in_dim, temp_n_classes)
    
    # 检查是否启用各种位置编码
    add_lap_pos_enc = temp_net_params.get('lap_pos_enc', False)
    add_metapath2vec_pe = temp_net_params.get('metapath2vec_pos_enc', False)
    metapath2vec_pe_paths = temp_net_params.get('metapath2vec_pe_paths', [])
    metapath_names = temp_net_params.get('metapath_names', [])
    pos_enc_dim = temp_net_params.get('pos_enc_dim', 8)
    
    data, edge_types = prepare_heterogeneous_data(
        args.data_path, 
        add_lap_pos_enc=add_lap_pos_enc, 
        pos_enc_dim=pos_enc_dim,
        add_metapath2vec_pe=add_metapath2vec_pe,
        metapath2vec_pe_paths=metapath2vec_pe_paths,
        metapath_names=metapath_names
    )
    
    # Move data to device
    data = data.to(device)
    
    # 获取数据维度信息
    in_dim = data.x.shape[1]  # 节点特征维度
    n_classes = len(torch.unique(data.y))  # 类别数
    
    print(f"节点特征维度: {in_dim}")
    print(f"类别数: {n_classes}")
    print(f"节点数: {data.x.size(0)}")
    print(f"边数: {data.edge_index.size(1)}")
    
    # 设置参数
    params = get_default_params()
    params['epochs'] = args.epochs
    params['init_lr'] = args.lr
    params['seed'] = args.seed
    
    net_params = get_default_net_params(device, in_dim, n_classes)
    net_params['hidden_dim'] = args.hidden_dim
    net_params['pe_fusion_mode'] = args.pe_fusion_mode
    
    print("训练参数:")
    print(json.dumps(params, indent=2))
    print("\n网络参数:")
    print(json.dumps({k: v for k, v in net_params.items() if k != 'device'}, indent=2))
    
    # 打印PE融合模式信息
    pe_fusion_mode = net_params.get('pe_fusion_mode', 'concat')
    print(f"\nPE融合模式: {pe_fusion_mode.upper()}")
    print("位置编码使用情况:")
    print(f"  - LaPe: {net_params.get('lap_pos_enc', False)}")
    print(f"  - WL PE: {net_params.get('wl_pos_enc', False)}")
    print(f"  - Metapath2vec PE: {net_params.get('metapath2vec_pos_enc', False)}")
    print(f"  - PE维度: {net_params.get('pos_enc_dim', 8)}")
    
    # 打印使用的元路径信息
    if hasattr(data, 'metapath_names'):
            print(f"  - 使用的元路径: {data.metapath_names}")
    print(f"  - 元路径PE文件: {data.metapath2vec_pe_paths if hasattr(data, 'metapath2vec_pe_paths') else []}")
    print(f"  - Expected number of heads: {1 + len(data.metapath_names)} (1 node feature head + {len(data.metapath_names)} PE heads)")
    
    # 创建基于元路径的多头注意力模型
    print("\n正在创建基于元路径的多头注意力模型...")
    model = MetapathGraphTransformerNet(net_params)
    model = model.to(device)
    
    # 注意：模型层将在第一次前向传播时动态创建
    print("注意：模型层将在第一次前向传播时动态创建")
    print("参数统计将在训练完成后进行")
    
    # Check if only run attention analysis
    if args.attention_analysis_only:
        print("\nAttention analysis only mode...")
        
        # Load pre-trained model (if exists)
        model_path = os.path.join(args.out_dir, 'latest_model.pt')
        if os.path.exists(model_path):
            print(f"Loading pre-trained model: {model_path}")
            model.load_state_dict(torch.load(model_path, map_location=device))
        else:
            print("Warning: No pre-trained model found, will use randomly initialized model for analysis")
        
        # Run attention analysis
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_dir = os.path.join(args.out_dir, f'attention_analysis_only_{timestamp}')
        os.makedirs(save_dir, exist_ok=True)
        
        analysis_results = run_attention_analysis(
            model, data, save_dir, device
        )
        
        if analysis_results:
            print(f"\nAttention analysis completed! Results saved to: {analysis_results['analysis_dir']}")
        else:
            print("\nAttention analysis failed!")
        
        return
    
    # 训练模型
    print("\n开始训练...")
    results = train_model(model, data, data.train_mask, data.val_mask, data.test_mask, 
                         params, net_params, device)
    
    # 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(args.out_dir, f'experiment_metapath_multihead_{timestamp}')
    save_results(results, model, params, net_params, save_dir, results['results_df'], data)
    
    # Run attention analysis (if enabled)
    if args.enable_attention_analysis:
        print("\nEnabling attention analysis...")
        analysis_results = run_attention_analysis(
            model, data, save_dir, device
        )
        
        if analysis_results:
            print(f"Attention analysis completed! Analysis results saved to: {analysis_results['analysis_dir']}")
        else:
            print("Attention analysis failed!")
    
    print("\nTraining completed!")
    print(f"Results saved to: {save_dir}")
    
    # Print attention analysis information
    if args.enable_attention_analysis:
        print(f"Attention analysis results saved to: {save_dir}/attention_analysis/")
        print("Contains the following files:")
        print("  - attention_score_heatmap.png: Attention score heatmap")
        print("  - attention_score_distribution.png: Attention score distribution")
        print("  - attention_entropy_heatmap.png: Attention entropy heatmap")
        print("  - attention_analysis_report.json: Detailed analysis report")
        print("  - attention_analysis_report.txt: Text analysis report")

if __name__ == "__main__":
    main() 