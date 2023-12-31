import os
import scipy.io
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2

class SkeletonDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.file_list = self._get_file_list()
        self.transform = transform

    def _get_file_list(self):
        file_list = []
        for root, _, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith('.mat') or file.endswith('.png'):
                    file_list.append(os.path.join(root, file))
        return file_list

    def __len__(self):
        return len(self.file_list) // 2  # 每对 .mat 和 .png 文件算作一个样本

    def __getitem__(self, idx):
        mat_file_path = self.file_list[idx * 2]
        png_file_path = self.file_list[idx * 2 + 1]

        # 使用 scipy.io 加载 .mat 文件
        mat_data = scipy.io.loadmat(mat_file_path)  # 根据实际数据结构进行调整

        # 使用 PIL 加载 .png 文件
        png_image = Image.open(png_file_path)
        png_image = np.array(png_image)  # 将图像转换为 NumPy 数组

        # 返回数据和标签（这里只是示例，你需要根据实际数据结构来定义）
        data = {
            'skeleton': mat_data['3D_skeleton_annotation'],
            'image': png_image,
        }

        if self.transform:
            # 转换数据为 PyTorch Tensor（根据需要）
            data = self._transform_data(data)

        return data

    def _transform_data(self, data):
        # 在这里可以添加数据预处理或转换操作
        # 例如，将 NumPy 数组转换为 PyTorch Tensor
        data['image'] = cv2.resize(data['image'], (224, 224))
        # 归一化
        data['image'] = cv2.normalize(data['image'], None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        data['image'] = torch.tensor(data['image'], dtype=torch.float32) 

        data['image'] = data['image'].unsqueeze(0)
        
        data['skeleton'] = torch.tensor(data['skeleton'], dtype=torch.float32)
        data['skeleton'] = data['skeleton'].reshape(72)
        return data


if __name__ == "__main__":
    dataset = SkeletonDataset(r'dataset\train',True)
    import cv2
    import numpy as np
    # for i in range(dataset.__len__()):
    #     sample = dataset[i]

    #     print(i, sample['skeleton'], sample['image'])
    #     if i >= 4:
    #         break

    batch_size = 8
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # 遍历 data_loader 来获取批次的数据
    for batch in data_loader:
        # 获取批次的数据和标签
        skeleton = batch['skeleton']
        image = batch['image']
        print(skeleton.shape)
        print(image.shape)

        for image_tensor in batch['image']:
            # 将张量从GPU移动到CPU（如果在GPU上）
            image_tensor_cpu = image_tensor.cpu()

            # 将PyTorch张量转换为NumPy数组
            image_array = image_tensor_cpu.numpy()

            # 将NumPy数组转换为OpenCV格式的图像（灰度或彩色）
            if image_array.shape[0] == 1:
                image_array = image_array.squeeze()  # 移除单通道的维度
                image = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)  # 转为彩色图像
            else:
                image = cv2.cvtColor(image_array.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)  # 转换通道顺序


            # 显示图像
            cv2.imshow('Image', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        break