import os
import scipy.io
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import h5py

class SkeletonDatasetHDF5(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.passed_idx = 0
        self.file_list = self._get_file_list()
        self.file_idx = 0
        # self.file_open = False  # 用于保存当前打开的HDF5文件
        # self.transform = transform

    def _get_file_list(self):
        file_list = {}
        for root, _, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith('.h5'):
                    file_list.append(os.path.join(root, file))

        return file_list

    def __len__(self):
        length = 0
        for file in self.file_list:
            with h5py.File(file, 'r') as f:
                length += f['skeleton']

        return length  # 每对 .mat 和 .png 文件算作一个样本

    def __getitem__(self, i):
        idx = i - self.passed_idx
        if self.f is None:
            self.f = h5py.File(self.file_list[self.file_idx], 'r')
            self.file_idx += 1
        # 返回数据和标签（这里只是示例，你需要根据实际数据结构来定义）
        if idx < f['skeleton'].__len__():
            data = {
                'skeleton': f['skeleton'][idx],
                'image': f['image'][idx],
                'gender': f['gender'][idx],
                'trans' : f['trans'][idx], # 根节点偏移量
            }
        else:
            self.passed_idx += idx
            data = {
                'skeleton': f['skeleton'][idx-1],
                'image': f['image'][idx-1],
                'gender': f['gender'][idx-1],
                'trans' : f['trans'][idx-1], # 根节点偏移量
            }
            self.f.close()


        return data
        

    # def _transform_data(self, data):
    #     # 在这里可以添加数据预处理或转换操作
    #     # 例如，将 NumPy 数组转换为 PyTorch Tensor

    #     '''
    #     pressurepose数据集最开始的.p文件中 skeleton, trans的保存格式为float64, images的为int8
    #     '''

    #     # 1. image预处理 归一化到0~1范围内 从int8转为float32
    #     data['image'] = cv2.resize(data['image'], (224, 224))
    #     # 归一化到0~1
    #     data['image'] = data['image'] / 255.0

    #     data['image'] = torch.tensor(data['image'], dtype=torch.float32) 

    #     data['image'] = data['image'].unsqueeze(0)
        
    #     # 2. skeleton预处理 从float64转为float32
    #     data['skeleton'] = torch.tensor(data['skeleton'], dtype=torch.float32)
    #     data['skeleton'] = data['skeleton'].reshape(72)
        
    #     # 3. trans预处理 从float64转为float32
    #     data['trans'] =  torch.tensor(data['trans'], dtype=torch.float32)
    #     data['trans'] =  data['trans'].reshape(3)

    #     # 4. genders预处理 转为bool
    #     data['gender'] = torch.tensor(data['gender'], dtype = torch.bool)

    #     return data


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
        gender = batch['gender'] 
        trans = batch['trans']
        print(skeleton.shape)
        print(image.shape)
        print(gender.shape)
        print(trans.shape)

        print(skeleton.dtype)
        print(image.dtype)
        print(gender.dtype)
        print(trans.dtype)

        # for image_tensor in batch['image']:
        #     # 将张量从GPU移动到CPU（如果在GPU上）
        #     image_tensor_cpu = image_tensor.cpu()

        #     # 将PyTorch张量转换为NumPy数组
        #     image_array = image_tensor_cpu.numpy()

        #     # 将NumPy数组转换为OpenCV格式的图像（灰度或彩色）
        #     if image_array.shape[0] == 1:
        #         image_array = image_array.squeeze()  # 移除单通道的维度
        #         image = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)  # 转为彩色图像
        #     else:
        #         image = cv2.cvtColor(image_array.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)  # 转换通道顺序


        #     # 显示图像
        #     cv2.imshow('Image', image)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()

        break