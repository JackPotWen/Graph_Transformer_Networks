import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import time
import os
import json

from HGT import HeterogeneousGraphTransformerNet, accuracy

def train_epoch(model, optimizer, device, data, train_mask, val_mask, test_mask):
    """训练一个epoch"""
    model.train()
    epoch_loss = 0
    epoch_train_acc = 0
    
    # 获取训练数据
    train_nodes = torch.where(train_mask)[0]
    
    # 创建训练子图
    from torch_geometric.utils import subgraph
    edge_index, edge_attr = subgraph(train_nodes, data.edge_index, data.edge_attr, 
                                    num_nodes=data.x.size(0), relabel_nodes=True)
    
    # 获取训练节点特征和标签
    batch_x = data.x[train_nodes].to(device)
    batch_edge_index = edge_index.to(device)
    batch_edge_attr = edge_attr.to(device) if edge_attr is not None else None
    batch_labels = data.y[train_nodes].to(device)
    
    # 创建训练数据对象
    from torch_geometric.data import Data
    train_data = Data(x=batch_x, edge_index=batch_edge_index, edge_attr=batch_edge_attr)
    train_data = train_data.to(device)
    
    optimizer.zero_grad()
    
    # 前向传播
    batch_scores = model.forward(train_data)
    
    # 计算损失
    loss = model.loss(batch_scores, batch_labels)
    loss.backward()
    optimizer.step()
    
    epoch_loss = loss.detach().item()
    epoch_train_acc = accuracy(batch_scores, batch_labels)
    
    return epoch_loss, epoch_train_acc, optimizer

def evaluate_network(model, device, data, mask, epoch):
    """评估网络"""
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    
    with torch.no_grad():
        # 获取评估数据
        eval_nodes = torch.where(mask)[0]
        
        # 创建评估子图
        from torch_geometric.utils import subgraph
        edge_index, edge_attr = subgraph(eval_nodes, data.edge_index, data.edge_attr, 
                                        num_nodes=data.x.size(0), relabel_nodes=True)
        
        # 获取评估节点特征和标签
        batch_x = data.x[eval_nodes].to(device)
        batch_edge_index = edge_index.to(device)
        batch_edge_attr = edge_attr.to(device) if edge_attr is not None else None
        batch_labels = data.y[eval_nodes].to(device)
        
        # 创建评估数据对象
        from torch_geometric.data import Data
        eval_data = Data(x=batch_x, edge_index=batch_edge_index, edge_attr=batch_edge_attr)
        eval_data = eval_data.to(device)
        
        # 前向传播
        batch_scores = model.forward(eval_data)
        
        # 计算损失和准确率
        loss = model.loss(batch_scores, batch_labels)
        epoch_loss = loss.detach().item()
        epoch_acc = accuracy(batch_scores, batch_labels)
        
    return epoch_loss, epoch_acc

def train_model(model, data, train_mask, val_mask, test_mask, params, net_params, device):
    """训练模型的主函数"""
    
    print("开始训练异构图Transformer模型...")
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
    
    # 学习率调度器（移除verbose参数以兼容不同PyTorch版本）
    try:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=params['lr_reduce_factor'],
            patience=params['lr_schedule_patience'], verbose=True
        )
    except TypeError:
        # 如果verbose参数不支持，使用默认设置
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=params['lr_reduce_factor'],
            patience=params['lr_schedule_patience']
        )
    
    # 记录训练过程
    epoch_train_losses, epoch_val_losses = [], []
    epoch_train_accs, epoch_val_accs = [], []
    best_val_acc = 0
    best_model_state = None
    
    # 训练循环
    try:
        with tqdm(range(params['epochs'])) as t:
            for epoch in t:
                t.set_description('Epoch %d' % epoch)
                start = time.time()
                
                # 训练
                epoch_train_loss, epoch_train_acc, optimizer = train_epoch(
                    model, optimizer, device, data, train_mask, val_mask, test_mask
                )
                
                # 验证
                epoch_val_loss, epoch_val_acc = evaluate_network(
                    model, device, data, val_mask, epoch
                )
                
                # 测试
                epoch_test_loss, epoch_test_acc = evaluate_network(
                    model, device, data, test_mask, epoch
                )
                
                # 记录结果
                epoch_train_losses.append(epoch_train_loss)
                epoch_val_losses.append(epoch_val_loss)
                epoch_train_accs.append(epoch_train_acc)
                epoch_val_accs.append(epoch_val_acc)
                
                # 学习率调度
                scheduler.step(epoch_val_loss)
                
                # 保存最佳模型
                if epoch_val_acc > best_val_acc:
                    best_val_acc = epoch_val_acc
                    best_model_state = model.state_dict().copy()
                
                # 更新进度条
                t.set_postfix(
                    time=time.time()-start, 
                    lr=optimizer.param_groups[0]['lr'],
                    train_loss=epoch_train_loss, 
                    val_loss=epoch_val_loss,
                    train_acc=epoch_train_acc, 
                    val_acc=epoch_val_acc,
                    test_acc=epoch_test_acc
                )
                
                # 早停检查
                if optimizer.param_groups[0]['lr'] < params['min_lr']:
                    print("\n!! LR SMALLER OR EQUAL TO MIN LR THRESHOLD.")
                    break
                    
                # 时间限制检查
                if time.time() - start > params['max_time'] * 3600:
                    print('-' * 89)
                    print("Max_time for training elapsed {:.2f} hours, so stopping".format(params['max_time']))
                    break
                    
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early because of KeyboardInterrupt')
    
    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"加载最佳模型，验证准确率: {best_val_acc:.4f}")
    
    # 最终评估
    final_test_loss, final_test_acc = evaluate_network(model, device, data, test_mask, epoch)
    final_train_loss, final_train_acc = evaluate_network(model, device, data, train_mask, epoch)
    
    print("=" * 50)
    print("训练完成！")
    print(f"最终测试准确率: {final_test_acc:.4f}")
    print(f"最终训练准确率: {final_train_acc:.4f}")
    print(f"最佳验证准确率: {best_val_acc:.4f}")
    print("=" * 50)
    
    return {
        'final_test_acc': final_test_acc,
        'final_train_acc': final_train_acc,
        'best_val_acc': best_val_acc,
        'train_losses': epoch_train_losses,
        'val_losses': epoch_val_losses,
        'train_accs': epoch_train_accs,
        'val_accs': epoch_val_accs
    }

def save_results(results, model, params, net_params, save_dir):
    """保存训练结果"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 保存模型
    torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pt'))
    
    # 保存结果
    results_file = os.path.join(save_dir, 'training_results.json')
    with open(results_file, 'w') as f:
        json.dump({
            'final_test_acc': results['final_test_acc'],
            'final_train_acc': results['final_train_acc'],
            'best_val_acc': results['best_val_acc'],
            'params': params,
            'net_params': {k: str(v) if not isinstance(v, (int, float, bool)) else v 
                          for k, v in net_params.items()}
        }, f, indent=2)
    
    print(f"结果已保存到: {save_dir}")

def view_model_param(model, net_params):
    """查看模型参数数量"""
    total_param = 0
    print("模型详情:\n")
    for param in model.parameters():
        total_param += np.prod(list(param.data.size()))
    print(f'模型总参数数量: {total_param}')
    return total_param 