import torch
import numpy as np
import torch.nn as nn
from model_gtn import GTN
from model_fastgtn import FastGTNs
import pickle
import argparse
import os
from torch_geometric.utils import add_self_loops
from sklearn.metrics import f1_score as sk_f1_score
from utils import init_seed, _norm, f1_score
import copy

def save_weights(Ws_list, save_path):
    """保存权重矩阵列表到文件"""
    # 确保目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 将权重转换为可保存的格式
    save_Ws = []
    for Ws in Ws_list:  # 遍历每个epoch的Ws
        epoch_Ws = []
        for W_list in Ws:  # 遍历每个W_list
            W_np = [w.detach().cpu().numpy() for w in W_list]
            epoch_Ws.append(W_np)
        save_Ws.append(epoch_Ws)
    
    # 保存到文件
    with open(save_path, 'wb') as f:
        pickle.dump(save_Ws, f)

if __name__ == '__main__':
    init_seed(seed=777)
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='GTN',
                        help='Model')
    parser.add_argument('--dataset', type=str, default='DBLP',
                        help='Dataset')
    parser.add_argument('--epoch', type=int, default=50,
                        help='Training Epochs')
    parser.add_argument('--node_dim', type=int, default=64,
                        help='hidden dimensions')
    parser.add_argument('--num_channels', type=int, default=2,
                        help='number of channels')
    parser.add_argument('--lr', type=float, default=0.02,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.001,
                        help='l2 reg')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='number of GT/FastGT layers')
    parser.add_argument('--runs', type=int, default=1,
                        help='number of runs')
    parser.add_argument("--channel_agg", type=str, default='concat')
    parser.add_argument("--remove_self_loops", action='store_true', help="remove_self_loops")
    parser.add_argument("--save_dir", type=str, default='metapath_data/weight',
                        help='Directory to save weights')

    args = parser.parse_args()
    print(args)

    epochs = args.epoch
    node_dim = args.node_dim
    num_channels = args.num_channels
    lr = args.lr
    weight_decay = args.weight_decay
    num_layers = args.num_layers

    with open('data/%s/node_features.pkl' % args.dataset,'rb') as f:
        node_features = pickle.load(f)
    with open('data/%s/edges.pkl' % args.dataset,'rb') as f:
        edges = pickle.load(f)
    with open('data/%s/labels.pkl' % args.dataset,'rb') as f:
        labels = pickle.load(f)

    num_nodes = edges[0].shape[0]
    args.num_nodes = num_nodes
    # build adjacency matrices for each edge type
    A = []
    for i,edge in enumerate(edges):
        edge_tmp = torch.from_numpy(np.vstack((edge.nonzero()[1], edge.nonzero()[0]))).type(torch.cuda.LongTensor)
        value_tmp = torch.ones(edge_tmp.shape[1]).type(torch.cuda.FloatTensor)
        A.append((edge_tmp,value_tmp))
    edge_tmp = torch.stack((torch.arange(0,num_nodes),torch.arange(0,num_nodes))).type(torch.cuda.LongTensor)
    value_tmp = torch.ones(num_nodes).type(torch.cuda.FloatTensor)
    A.append((edge_tmp,value_tmp))
    
    num_edge_type = len(A)
    node_features = torch.from_numpy(node_features).type(torch.cuda.FloatTensor)
    train_node = torch.from_numpy(np.array(labels[0])[:,0]).type(torch.cuda.LongTensor)
    train_target = torch.from_numpy(np.array(labels[0])[:,1]).type(torch.cuda.LongTensor)
    valid_node = torch.from_numpy(np.array(labels[1])[:,0]).type(torch.cuda.LongTensor)
    valid_target = torch.from_numpy(np.array(labels[1])[:,1]).type(torch.cuda.LongTensor)
    test_node = torch.from_numpy(np.array(labels[2])[:,0]).type(torch.cuda.LongTensor)
    test_target = torch.from_numpy(np.array(labels[2])[:,1]).type(torch.cuda.LongTensor)
    num_classes = np.max([torch.max(train_target).item(), torch.max(valid_target).item(), torch.max(test_target).item()])+1

    # initialize a model
    model = GTN(num_edge=len(A),
                num_channels=num_channels,
                w_in = node_features.shape[1],
                w_out = node_dim,
                num_class=num_classes,
                num_layers=num_layers,
                num_nodes=num_nodes,
                args=args)        

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.cuda()
    loss = nn.CrossEntropyLoss()
    # 创建保存目录
    save_dir = os.path.join(args.save_dir, args.dataset)
    os.makedirs(save_dir, exist_ok=True)
    
    best_val_f1 = 0
    best_epoch = 0
    all_Ws = []  # 存储所有epoch的权重矩阵
    
    for i in range(epochs):
        model.zero_grad()
        model.train()
        loss_train, y_train, Ws = model(A, node_features, train_node, train_target)
        
        train_f1 = torch.mean(f1_score(torch.argmax(y_train.detach(),dim=1), train_target, num_classes=num_classes)).cpu().numpy()
        sk_train_f1 = sk_f1_score(train_target.detach().cpu(), np.argmax(y_train.detach().cpu(), axis=1), average='micro')
        
        loss_train.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_loss, y_valid, _ = model(A, node_features, valid_node, valid_target)
            test_loss, y_test, _ = model(A, node_features, test_node, test_target)
            
            val_f1 = torch.mean(f1_score(torch.argmax(y_valid,dim=1), valid_target, num_classes=num_classes)).cpu().numpy()
            sk_val_f1 = sk_f1_score(valid_target.detach().cpu(), np.argmax(y_valid.detach().cpu(), axis=1), average='micro')
            
            test_f1 = torch.mean(f1_score(torch.argmax(y_test,dim=1), test_target, num_classes=num_classes)).cpu().numpy()
            sk_test_f1 = sk_f1_score(test_target.detach().cpu(), np.argmax(y_test.detach().cpu(), axis=1), average='micro')
            
            # 保存当前epoch的权重矩阵
            all_Ws.append(Ws)
            
            # 更新最佳验证集性能
            if sk_val_f1 > best_val_f1:
                best_val_f1 = sk_val_f1
                best_epoch = i + 1
                best_Ws = Ws  # 保存最佳epoch的权重矩阵
            
            print('Epoch: {:04d}'.format(i+1),
                  'loss_train: {:.4f}'.format(loss_train.item()),
                  'train_f1: {:.4f}'.format(train_f1),
                  'val_f1: {:.4f}'.format(val_f1),
                  'test_f1: {:.4f}'.format(test_f1))
    
    # 保存所有epoch的权重矩阵
    save_weights(all_Ws, os.path.join(save_dir, 'all_weights.pkl'))
    print(f'Saved all weights to {os.path.join(save_dir, "all_weights.pkl")}')
    
    # 保存最佳epoch的权重矩阵
    save_weights([best_Ws], os.path.join(save_dir, 'best_weights.pkl'))
    print(f'Saved best weights to {os.path.join(save_dir, "best_weights.pkl")}')

    print('Training completed!')
    print(f'Best validation F1: {best_val_f1:.4f} at epoch {best_epoch}') 