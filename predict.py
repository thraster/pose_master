import torch
from vgg import VGG16
import os
from PIL import Image
import cv2
import numpy as np

def predict_vgg16(img_path = 'predict_folder', checkpoint_path = None, ):
    """
    加载预训练模型参数并使用模型进行预测。

    参数：
    - model: 未初始化的模型实例
    - model_path: 模型参数.pth文件的路径
    - input_data: 输入数据（Tensor或Numpy数组）

    返回值：
    - predictions: 模型的预测结果
    """
    # 加载模型参数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VGG16().to(device)  # 根据实际情况调整模型初始化方式

    if checkpoint_path != None:
        checkpoint = torch.load(checkpoint_path)
        total_epoch = checkpoint['epochs']
        print(f"loading checkpoint [{checkpoint_path}] successed!")
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"loading state_dict successed!")
        print(f"checkpoint info: epoch = {checkpoint['epochs']}")
    else:
        print('need to specify the checkpoint path to load the parameters!')
        print('model running with random parameters now...')

    print('model initialized on', device)


    # 将模型设置为评估模式
    model.eval()

    image_list = []

    # 将输入数据转换为Tensor
    for root, _, files in os.walk(img_path):
        for file in files:
            if file.endswith('.png'):
                input_data = Image.open(os.path.join(root, file))

                input_data = np.array(input_data)

                input_data = cv2.resize(input_data, (224, 224))
                # 归一化
                input_data = cv2.normalize(input_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

                input_data = torch.tensor(input_data, dtype=torch.float32) 

                input_data = input_data.unsqueeze(0).unsqueeze(0)
                image_list.append(input_data)
                # input_data.to(device)
    image = torch.cat(image_list, dim=0)
    print(image.shape)
    # print(image[16])
    # 执行模型预测
    with torch.no_grad():
        output = model(image.to(device))

    print(output)
    print(output.shape)

    return output

    # 如果模型有多个输出，你可以根据需要返回其中一个或多个
    # 这里只返回第一个输出
    # predictions = output[0]

    # return predictions

# 使用示例
if __name__ == "__main__":
    # 设置模型文件的路径
    model_path = r'checkpoints/best_epoch45.pth'  # 替换成你的.pth文件路径

    # 准备输入数据（示例：随机生成一个大小为[1, 3, 224, 224]的张量）
    # input_data = torch.randn(1, 3, 224, 224)

    # 调用函数进行预测
    # predictions = 
    
    output = predict_vgg16(checkpoint_path = model_path)

    
    # 打印预测结果
    # print(predictions)
