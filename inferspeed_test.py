import torch
import time
from models.mobilenet_smpl_batch import mobilenet  # 以ResNet-18为例
from models.mobilenet_smpl import mobilenet as mobilenet_itersmpl
from models.posenet_res import posenet as posenet_itersmpl
from models.resnet_smpl_batch import posenet

batch_size = 8
input_data = torch.randn(batch_size, 1, 224, 224)
gender = torch.randint(2, (batch_size, 1))
device = 'cpu'
# 创建模型实例
model1 = mobilenet_itersmpl(device=device).to(device)
model2 = posenet_itersmpl(device=device).to(device)

def inference_time(model, input_data, gender, num_iterations=100, device = 'cpu'):
    total_time = 0
    print(f"infering, model = {model.name}, for {num_iterations} iters")
    with torch.no_grad():
        for _ in range(num_iterations):
            start_time = time.time()
            output = model(input_data.to(device), gender.to(device))
            end_time = time.time()
            total_time += end_time - start_time
    return total_time / num_iterations

num_iterations = 128
time1 = inference_time(model1, input_data, gender, num_iterations, device=device)
time2 = inference_time(model2, input_data, gender, num_iterations, device=device)

print(f"MobileNet Inference avg Time: {time1} seconds, on [{device}]")
print(f"PoseNet Inference avg Time: {time2} seconds, on [{device}]")

'''
infering, model = mobilenetv2_smpl, for 128 iters
infering, model = resnet+smpl, for 128 iters
MobileNet_itersmpl Inference avg Time: 0.5457477737218142 seconds, on [cpu]
PoseNet_itersmpl Inference avg Time: 0.44180510006845 seconds, on [cpu]
MobileNet_itersmpl Inference avg Time: 0.2886335998773575 seconds, on [cuda]
PoseNet_itersmpl Inference avg Time: 0.2625382486730814 seconds, on [cuda]


# 多跑点iter
infering, model = mobilenetv2_smpl, for 128 iters
infering, model = resnet18_smpl, for 128 iters
MobileNet Inference avg Time: 0.5178092755377293 seconds, on [cpu]
PoseNet Inference avg Time: 0.42258534394204617 seconds, on [cpu]
MobileNet Inference avg Time: 0.13631493411958218 seconds, on [cuda]
PoseNet Inference avg Time: 0.10320301167666912 seconds, on [cuda]

'''