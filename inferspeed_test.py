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
model1 = mobilenet(device=device).to(device)
model2 = posenet(device=device).to(device)

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
MobileNet_itersmpl Inference avg Time: 0.7484717845916748 seconds, on [cpu]
PoseNet_itersmpl Inference avg Time: 0.5922966480255127 seconds, on [cpu]
MobileNet_itersmpl Inference avg Time: 0.6717566728591919 seconds, on [cuda]
PoseNet_itersmpl Inference avg Time: 0.4732471227645874 seconds, on [cuda]


# 多跑点iter
MobileNet Inference avg Time: 0.6836470842361451 seconds, on [cpu]
PoseNet Inference avg Time: 0.570149278640747 seconds, on [cpu]
MobileNet Inference avg Time: 0.35993931293487547 seconds, on [cuda]
PoseNet Inference avg Time: 0.16239492893218993 seconds, on [cuda]

'''