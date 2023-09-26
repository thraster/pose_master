import scipy.io

import cv2
import torch
# import torchvision.transforms as transforms

def load_png(image_path):
    try:
        # 使用 OpenCV 读取 PNG 图像并转换为灰度图像
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            raise FileNotFoundError(f"无法读取图像文件: {image_path}")
        
        # 调整图像大小为 [1, 224, 224]
        image = cv2.resize(image, (224, 224))

        # 将图像数据归一化到 [0, 1] 范围
        image = image.astype(float) / 255.0

        # 将 NumPy 数组转换为 PyTorch Tensor
        image_tensor = torch.tensor(image, dtype=torch.float32)
        
        # 如果需要，可以添加任何其他的预处理操作
        # # 将通道扩展为3
        # image_tensor = image_tensor.expand(3, -1, -1) 

        # 在 PyTorch 中图像通常具有形状 (H, W)，需要添加通道维度 (C)
        # image_tensor = image_tensor.unsqueeze(0)  # 添加通道维度 (C=1)
        
        print(image_tensor, image_tensor.shape)
        return image_tensor
    
    except Exception as e:
        print(f"加载图像出错: {str(e)}")
        return None



def load_mat(path):
    # 使用 scipy.io.loadmat 加载 .mat 文件
    mat_data = scipy.io.loadmat(path)

    # 从 mat_data 中提取您需要的数据，假设您想要提取名为 'your_variable' 的变量
    your_variable = mat_data['3D_skeleton_annotation']

    # 将 NumPy 数组转换为 PyTorch Tensor
    your_variable_tensor = torch.tensor(your_variable)

    # 现在，your_variable_tensor 包含了 .mat 文件中的数据，并以 PyTorch Tensor 格式存储
    print(your_variable_tensor, your_variable_tensor.shape)

    return your_variable_tensor






if __name__=="__main__":
    path = r'dataset\test\test_roll0_f_lay_set14_1500\0.mat'
    load_mat(path)
    
    load_png(r'dataset\test\test_roll0_f_lay_set14_1500\0.png')