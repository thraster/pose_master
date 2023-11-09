import torch
from thop import profile
from models.mobilenet_smpl_batch import mobilenet  # 以ResNet-18为例
from models.mobilenet_smpl import mobilenet as mobilenet_itersmpl
from models.posenet_res import posenet as posenet_itersmpl
from models.resnet_smpl_batch import posenet
# FLOPs更适合评估通用深度学习模型

batch_size = 8
# 创建一个示例输入张量
input_data = torch.randn(batch_size, 1, 224, 224)  # （batch_size, channels, height, width）
gender = torch.randint(2,(batch_size,1))
# 加载模型
model1 = mobilenet(device='cpu')
model2 = posenet(device='cpu')

model3 = mobilenet_itersmpl(device='cpu')
model4 = posenet_itersmpl(device='cpu')


# 使用thop库的profile函数来估算FLOPs
flops1, params = profile(model1, inputs=(input_data,gender))
flops2, params = profile(model2, inputs=(input_data,gender))
flops3, params = profile(model3, inputs=(input_data,gender))
flops4, params = profile(model4, inputs=(input_data,gender))

print(f"batch size = {batch_size}")
print(f"mobilenet FLOPs: {flops1 / 1e9} GigaFLOPs")  # 转换单位为GigaFLOPs
print(f"resnet FLOPs: {flops2 / 1e9} GigaFLOPs")  # 转换单位为GigaFLOPs
print(f"mobilenet_itersmpl FLOPs: {flops3 / 1e9} GigaFLOPs")  # 转换单位为GigaFLOPs
print(f"resnet_itersmpl FLOPs: {flops4 / 1e9} GigaFLOPs")  # 转换单位为GigaFLOPs


'''
---------------------------------------------
Tested on PC with windows10, pytorch1.5
NVIDIA GeForce GTX TITAN X @12GB GPU RAM
Intel Xeon CPU E5-2623 v3 @3.00GHz, 16GB RAM
---------------------------------------------


batch size = 8
mobilenet FLOPs: 2.553233408 GigaFLOPs
resnet FLOPs: 13.959114752 GigaFLOPs
mobilenet_itersmpl FLOPs: 2.553233408 GigaFLOPs
resnet_itersmpl FLOPs: 13.959114752 GigaFLOPs
# 参数少可能轻量结构多，反而计算过程多些，速度慢些

'''