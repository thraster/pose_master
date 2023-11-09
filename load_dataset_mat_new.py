import os
import scipy.io
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2

class SkeletonDatasetMAT(Dataset):
    def __init__(self, root_dir, transform=False):
        self.root_dir = root_dir
        self.file_list = self._get_file_list()
        self.transform = transform

    def _get_file_list(self):
        file_list = []
        for root, _, files in os.walk(self.root_dir):
            for file in files:
                
                if file.endswith('.mat'):
                    file_list.append(os.path.join(root, file))
        return file_list

    def __len__(self):
        return len(self.file_list)  # 每个 .mat 算作一个样本

    def __getitem__(self, idx):
        mat_file_path = self.file_list[idx]
       
        # 使用 scipy.io 加载 .mat 文件
        mat_data = scipy.io.loadmat(mat_file_path)  # 根据实际数据结构进行调整
        
        # return mat_data
        
        # print(mat_data.keys())
        '''
        dict_keys(['__header__', '__version__', '__globals__', 
        'image', 'mesh_contact', 'mesh_depth', 'skeleton',
          'trans', 'shape', 'pose', 'gender']
        '''
        '''
        mat_data的格式
        Key: skeleton, Shape: torch.Size([1, 72]), Dtype: torch.float64
        Key: image, Shape: torch.Size([1, 1728]), Dtype: torch.int8
        Key: gender, Shape: torch.Size([1, 1]), Dtype: torch.int32
        Key: trans, Shape: torch.Size([1, 3]), Dtype: torch.float64
        Key: shape, Shape: torch.Size([1, 10]), Dtype: torch.float64
        Key: pose, Shape: torch.Size([1, 72]), Dtype: torch.float64
        Key: mesh_depth, Shape: torch.Size([64, 27]), Dtype: torch.int32
        Key: mesh_contact, Shape: torch.Size([64, 27]), Dtype: torch.uint8
        '''
        # 返回数据和标签（这里只是示例，你需要根据实际数据结构来定义）
        data = {
            'skeleton': mat_data['skeleton'].reshape(24,3),
            'image': mat_data['image'].reshape(64,27),
            'gender': mat_data['gender'].item(),
            'trans' : mat_data['trans'].squeeze(0), # 根节点偏移量
            'shape' : mat_data['shape'].squeeze(0),
            'pose' : mat_data['pose'].squeeze(0),
            'mesh_depth' : mat_data['mesh_depth'],
            'mesh_contact' : mat_data['mesh_contact']
        }

        
     
        # # 做维度的调整等
        # if self.transform:
        #     # 转换数据为 PyTorch Tensor（根据需要）
        #     data = self._transform_data(data)
        # return mat_data
        return data

    # def _transform_data(self, data):
        
    #     # 在这里可以添加数据预处理或转换操作
    #     '''
    #     转为期望的格式
    #     Key: skeleton, Shape: torch.Size([24, 3]), Dtype: torch.float64
    #     Key: image, Shape: torch.Size([64, 27]), Dtype: torch.uint8
    #     Key: gender, <class 'int'>
    #     Key: trans, Shape: torch.Size([1, 3]), Dtype: torch.float64
        
    #     '''
    #     # '''
    #     # pressurepose数据集最开始的.p文件中 skeleton, trans的保存格式为float64, images的为int8
    #     # '''

    #     # 1. image [1, 72]int8到[64, 27]int8
    #     data['image'] = data['image'].view(64,27)
        
    #     # 2. skeleton预处理
    #     data['skeleton'] = data['skeleton'].reshape(24,3)

    #     # 3. genders
    #     data['gender'] = data['gender'][0,0]

    #     return data


if __name__ == "__main__":
    dataset = SkeletonDatasetMAT(r'D:\workspace\python_ws\bodies-at-rest-master\dataset\train')
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
        # print(batch.values())
        # print(batch.keys())
        skeleton = batch['skeleton']
        image = batch['image']
        gender = batch['gender'] 
        trans = batch['trans']
        shape = batch['shape']
        pose = batch['pose']

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