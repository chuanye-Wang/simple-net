import torch as tc
import pickle

# 假设 pkl 文件包含一个训练好的 PyTorch 模型
pkl_file_path = 'sample_weight.pkl'

# 使用 pickle 加载模型
with open(pkl_file_path, 'rb') as f:
    model = pickle.load(f)

# 打印模型结构
print(model)
