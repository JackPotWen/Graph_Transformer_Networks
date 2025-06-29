import pickle
import os
import numpy as np

# 加载标签文件
data_path = "../../data/DBLP"
with open(os.path.join(data_path, "labels.pkl"), 'rb') as f:
    labels = pickle.load(f)

print("标签类型:", type(labels))
print("标签长度:", len(labels) if hasattr(labels, '__len__') else "无长度属性")

if isinstance(labels, list):
    print("列表内容:")
    for i, item in enumerate(labels):
        print(f"  项目 {i}: 类型={type(item)}")
        if isinstance(item, list):
            print(f"    子列表长度: {len(item)}")
            if len(item) > 0:
                print(f"    第一个元素类型: {type(item[0])}")
                if hasattr(item[0], 'shape'):
                    print(f"    第一个元素形状: {item[0].shape}")
                if len(item) > 1:
                    print(f"    第二个元素类型: {type(item[1])}")
                    if hasattr(item[1], 'shape'):
                        print(f"    第二个元素形状: {item[1].shape}")
        elif hasattr(item, 'shape'):
            print(f"    形状: {item.shape}")
        elif hasattr(item, 'dtype'):
            print(f"    数据类型: {item.dtype}")
        
        # 尝试转换为numpy数组查看形状
        try:
            if isinstance(item, list):
                arr = np.array(item)
                print(f"    转换为numpy数组后形状: {arr.shape}")
            else:
                arr = np.array(item)
                print(f"    转换为numpy数组后形状: {arr.shape}")
        except:
            print(f"    无法转换为numpy数组")
        print()

elif isinstance(labels, dict):
    print("字典键:", list(labels.keys()))
    for key, value in labels.items():
        print(f"  键 '{key}': 类型={type(value)}")
        if hasattr(value, 'shape'):
            print(f"    形状: {value.shape}")
        if hasattr(value, 'dtype'):
            print(f"    数据类型: {value.dtype}")
        if len(str(value)) < 200:
            print(f"    内容: {value}")
        print()

else:
    print("其他类型，尝试打印:")
    print(labels) 