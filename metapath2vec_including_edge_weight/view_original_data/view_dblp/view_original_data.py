import pickle
import numpy as np
import torch
import os

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
    # 获取当前文件的绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 获取项目根目录
    root_dir = os.path.abspath(os.path.join(current_dir, "../../../"))
    # 设置数据路径
    data_dir = os.path.join(root_dir, "data", "DBLP")
    
    print(f"数据目录: {data_dir}")
    
    # 要查看的文件列表
    files = ['edges.pkl', 'labels.pkl', 'node_features.pkl']
    
    for file in files:
        file_path = os.path.join(data_dir, file)
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            print_data_info(data, file)
        except Exception as e:
            print(f"\n处理文件 {file} 时出错:")
            print(f"错误信息: {str(e)}")

if __name__ == "__main__":
    main() 