import os
import json
import subprocess
import itertools
from datetime import datetime
import torch
import numpy as np

def run_single_experiment(params, experiment_id, log_dir):
    """运行单次实验"""
    print(f"\n开始实验 {experiment_id}")
    print("参数配置:", json.dumps(params, indent=2, ensure_ascii=False))
    
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 构建命令
    base_args = [
        'python', os.path.join(current_dir, 'train_gcn.py'),
        '--dataset', 'DBLP',
        '--save_dir', os.path.join(os.path.dirname(current_dir), 'my_gtn/metapath_data'),
        '--hidden_channels', str(params['hidden_channels']),
        '--epochs', str(params['epochs']),
        '--lr', str(params['lr']),
        '--weight_decay', str(params['weight_decay']),
        '--threshold', str(params['threshold']),
        '--seed', str(params['seed']),
        '--experiment_id', str(experiment_id),  # 添加实验ID
        '--log_dir', log_dir  # 添加日志目录
    ]
    
    cmd = ' '.join(base_args)
    print(f"\n运行命令:\n{cmd}")
    
    try:
        # 运行训练
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        
        # 解析输出获取最佳结果
        best_val_f1 = 0.0
        best_test_f1 = 0.0
        for line in result.stdout.split('\n'):
            if 'Best Validation F1' in line:
                best_val_f1 = float(line.split(': ')[-1])
            elif 'Best Test F1' in line:
                best_test_f1 = float(line.split(': ')[-1])
        
        # 记录实验结果
        experiment_result = {
            'experiment_id': experiment_id,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'parameters': params,
            'best_validation_f1': best_val_f1,
            'best_test_f1': best_test_f1,
            'status': 'success'
        }
        
        print(f"\n实验 {experiment_id} 完成:")
        print(f"最佳验证集F1: {best_val_f1:.4f}")
        print(f"最佳测试集F1: {best_test_f1:.4f}")
        
        return experiment_result
        
    except subprocess.CalledProcessError as e:
        print(f"\n实验 {experiment_id} 失败: {e}")
        experiment_result = {
            'experiment_id': experiment_id,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'parameters': params,
            'status': 'failed',
            'error': str(e)
        }
        return experiment_result

def run_experiments():
    """运行超参数搜索实验"""
    print("="*50)
    print("开始超参数搜索实验...")
    print("="*50)
    
    # 创建日志目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(current_dir, 'experiment_logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # 定义超参数搜索空间
    param_grid = {
        'hidden_channels': [32, 64, 128],
        'epochs': [100, 200, 300],
        'lr': [0.001, 0.01, 0.05],
        'weight_decay': [1e-5, 5e-4, 1e-3],
        'threshold': [0.05, 0.1, 0.2],
        'seed': [42]  # 固定随机种子
    }
    
    # 生成所有参数组合
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(itertools.product(*param_values))
    
    # 存储所有实验结果
    all_results = []
    best_result = None
    best_f1 = 0.0
    
    # 运行所有实验
    for i, param_values in enumerate(param_combinations):
        params = dict(zip(param_names, param_values))
        experiment_id = f"exp_{i+1:03d}"
        
        # 运行实验
        result = run_single_experiment(params, experiment_id, log_dir)
        all_results.append(result)
        
        # 更新最佳结果
        if result['status'] == 'success' and result['best_validation_f1'] > best_f1:
            best_f1 = result['best_validation_f1']
            best_result = result
        
        # 保存当前所有结果
        with open(os.path.join(log_dir, 'all_experiments.json'), 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        # 保存最佳结果
        if best_result:
            with open(os.path.join(log_dir, 'best_experiment.json'), 'w', encoding='utf-8') as f:
                json.dump(best_result, f, indent=2, ensure_ascii=False)
    
    # 打印实验总结
    print("\n" + "="*50)
    print("实验总结:")
    print(f"总实验数: {len(all_results)}")
    print(f"成功实验数: {sum(1 for r in all_results if r['status'] == 'success')}")
    print(f"失败实验数: {sum(1 for r in all_results if r['status'] == 'failed')}")
    
    if best_result:
        print("\n最佳实验结果:")
        print(f"实验ID: {best_result['experiment_id']}")
        print(f"最佳验证集F1: {best_result['best_validation_f1']:.4f}")
        print(f"最佳测试集F1: {best_result['best_test_f1']:.4f}")
        print("\n最佳参数配置:")
        for param, value in best_result['parameters'].items():
            print(f"{param}: {value}")
    
    print("\n" + "="*50)
    print(f"所有实验结果已保存到: {log_dir}")
    print("="*50)

if __name__ == "__main__":
    run_experiments() 