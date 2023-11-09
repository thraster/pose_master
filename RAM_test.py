import torch
# from torch.utils.bottleneck import peak_mem_metric
from models.mobilenet_smpl_batch import mobilenet  # 以ResNet-18为例
from models.mobilenet_smpl import mobilenet as mobilenet_itersmpl
from models.posenet_res import posenet as posenet_itersmpl
from models.resnet_smpl_batch import posenet
'''
还没写出来...


'''
batch_size = 8
# 创建一个示例输入张量
input_data = torch.randn(batch_size, 1, 224, 224)  # （batch_size, channels, height, width）
gender = torch.randint(2,(batch_size,1))
# 加载模型
model1 = mobilenet(device='cpu')
model2 = posenet(device='cpu')

model3 = mobilenet_itersmpl(device='cpu')
model4 = posenet_itersmpl(device='cpu')

# with torch.cuda.amp.autocast():
output = model1(input_data)

memory_stats = torch.cuda.memory_stats(device='cpu')
print(f"Memory usage (current): {memory_stats['allocated_bytes.all.current'] / 1024 / 1024:.2f} MB")
print(f"Memory usage (max): {memory_stats['allocated_bytes.all.peak'] / 1024 / 1024:.2f} MB")