import subprocess
import os

def run_gtn():
    # 创建保存目录
    os.makedirs('metapath_data/DBLP', exist_ok=True)
    
    # 构造命令
    command = [
        'python', 'train.py',
        '--dataset', 'DBLP',
        '--model', 'GTN',
        '--epochs', '200',
        '--node_dim', '64',
        '--num_channels', '2',
        '--lr', '0.01',
        '--weight_decay', '0.001',
        '--num_layers', '1',
        '--seed', '42',
        '--save_metapath',
        '--save_dir', 'metapath_data'
    ]
    
    # 运行命令
    print("开始训练GTN模型并保存metapath结构...")
    subprocess.run(command)
    print("训练完成！metapath结构已保存到 metapath_data/DBLP 目录")

if __name__ == '__main__':
    run_gtn() 