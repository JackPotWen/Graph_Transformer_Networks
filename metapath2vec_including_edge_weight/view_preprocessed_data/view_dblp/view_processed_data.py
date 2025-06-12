import pickle
import numpy as np
import torch
import os
import argparse

def print_data_info(data, name):
    print(f"\n{'='*50}")
    print(f"数据文件: {name}")
    print(f"{'='*50}")
    
    if isinstance(data, dict):
        print("\n数据结构 (字典):")
        for key, value in data.items():
            print(f"\n键: {key}")
            if isinstance(value, (np.ndarray, torch.Tensor)):
                print(f"类型: {type(value)}")
                print(f"形状: {value.shape}")
                print(f"数据类型: {value.dtype}")
                print(f"数值范围: [{value.min()}, {value.max()}]")
                if len(value.shape) == 1:
                    print(f"前5个元素: {value[:5]}")
            elif isinstance(value, dict):
                print(f"类型: 字典")
                print(f"包含的键: {list(value.keys())}")
            else:
                print(f"类型: {type(value)}")
                print(f"值: {value}")
    elif isinstance(data, (np.ndarray, torch.Tensor)):
        print(f"\n类型: {type(data)}")
        print(f"形状: {data.shape}")
        print(f"数据类型: {data.dtype}")
        print(f"数值范围: [{data.min()}, {data.max()}]")
        if len(data.shape) == 1:
            print(f"前5个元素: {data[:5]}")
    else:
        print(f"\n类型: {type(data)}")
        print(f"值: {data}")

def main():
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='查看处理后的DBLP数据集信息')
    parser.add_argument('--epoch', type=int, required=True, help='要查看的epoch编号')
    args = parser.parse_args()
    
    # 获取当前文件的绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 获取项目根目录
    root_dir = os.path.abspath(os.path.join(current_dir, "../../../"))
    # 设置数据路径
    data_dir = os.path.join(root_dir, "metapath_data", "H", "DBLP", "H_Epoch")
    
    # 构建文件名
    file_name = f"metapath_H_Epoch{args.epoch}.pkl"
    file_path = os.path.join(data_dir, file_name)
    
    print(f"正在查看 Epoch {args.epoch} 的数据")
    print(f"数据文件: {file_path}")
    
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        print_data_info(data, file_name)
    except Exception as e:
        print(f"\n处理文件 {file_name} 时出错:")
        print(f"错误信息: {str(e)}")

if __name__ == "__main__":
    main() 