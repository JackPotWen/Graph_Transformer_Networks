"""
    异构图Transformer主运行文件
    用于DBLP数据集的节点分类任务
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
from HGT import HeterogeneousGraphTransformerNet
from train_utils import train_model, save_results, view_model_param

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
        'epochs': 200,
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
        'n_heads': 8,  # 注意力头数
        'in_feat_dropout': 0.1,  # 输入特征dropout
        'dropout': 0.1,  # dropout
        'layer_norm': False,  # 是否使用层归一化
        'batch_norm': True,  # 是否使用批归一化
        'residual': True,  # 是否使用残差连接
        'edge_feat_dim': 1,  # 边特征维度
        'readout': 'mean',  # 读出方式
        'lap_pos_enc': False,  # 是否使用拉普拉斯位置编码
        'wl_pos_enc': False,  # 是否使用WL位置编码
        'pos_enc_dim': 8,  # 位置编码维度
        'node_feat_is_int': False,  # 节点特征是否为整数
        'device': device,
        'in_dim': in_dim,
        'n_classes': n_classes,
    }
    return net_params

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='异构图Transformer训练')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
    parser.add_argument('--use_gpu', action='store_true', help='是否使用GPU')
    parser.add_argument('--data_path', type=str, default='F:/github/Graph_Transformer_Networks/data/DBLP', help='数据路径')
    parser.add_argument('--out_dir', type=str, default='./results', help='输出目录')
    parser.add_argument('--epochs', type=int, default=200, help='训练轮数')
    parser.add_argument('--hidden_dim', type=int, default=128, help='隐藏层维度')
    parser.add_argument('--n_heads', type=int, default=8, help='注意力头数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # GPU设置
    device = gpu_setup(args.use_gpu, args.gpu_id)
    
    print("=" * 60)
    print("异构图Transformer训练")
    print("=" * 60)
    
    # 加载和准备数据
    print("正在加载数据...")
    data, edge_types = prepare_heterogeneous_data(args.data_path)
    
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
    net_params['n_heads'] = args.n_heads
    
    print("训练参数:")
    print(json.dumps(params, indent=2))
    print("\n网络参数:")
    print(json.dumps({k: v for k, v in net_params.items() if k != 'device'}, indent=2))
    
    # 创建模型
    print("\n正在创建模型...")
    model = HeterogeneousGraphTransformerNet(net_params)
    model = model.to(device)
    
    # 查看模型参数
    total_param = view_model_param(model, net_params)
    net_params['total_param'] = total_param
    
    # 训练模型
    print("\n开始训练...")
    results = train_model(model, data, data.train_mask, data.val_mask, data.test_mask, 
                         params, net_params, device)
    
    # 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(args.out_dir, f'experiment_{timestamp}')
    save_results(results, model, params, net_params, save_dir)
    
    print("\n训练完成！")
    print(f"结果保存在: {save_dir}")

if __name__ == "__main__":
    main() 