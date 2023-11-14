import h5py
import cv2
import torch
import numpy as np
import os
import lmdb
import sys
import pickle

# define the.
min_shape = -3
max_shape = 3
min_pose = -2.8452
max_pose = 4.1845
min_trans = -0.0241
max_trans = 1.5980

class SkeletonDatasetLMDB:
    def __init__(self, lmdb_path, transform = True):
        self.lmdb_path = lmdb_path
        self.env = lmdb.open(lmdb_path, readonly=True)
        self.transform = transform

    def __len__(self):
        with self.env.begin() as txn:
            return txn.stat()['entries']

    def __getitem__(self, index):
        with self.env.begin() as txn:
            data_bytes = txn.get(str(index).encode('ascii'))
        if data_bytes is None:
            raise IndexError(f"Index {index} not found in LMDB dataset.")
        data = pickle.loads(data_bytes)

        if self.transform:
                # 转换数据为 PyTorch Tensor（根据需要）
                data = self._transform_data(data)

        return data

    def _transform_data(self, data):
        # 在这里可以添加数据预处理或转换操作
        # 例如，将 NumPy 数组转换为 PyTorch Tensor

        '''
        pressurepose数据集最开始的.p文件中 skeleton, trans的保存格式为float64, images的为int8
        '''

        # 1. image预处理 归一化到0~1范围内 从int8转为uint8再转为float32
        data['image'] = cv2.resize((data['image']* 2).numpy().astype(np.uint8), (224, 224))
        # 归一化到0~1
        data['image'] = torch.tensor(data['image'], dtype=torch.float32)
        data['image'] = data['image'] / 255.0

        data['image'] = data['image'].unsqueeze(0)
        
        # 2. skeleton预处理 从float64转为float32
        data['skeleton'] = data['skeleton'].to(dtype=torch.float32)
        data['skeleton'] = data['skeleton'].reshape(72)
        
        # 3. trans预处理 从float64转为float32
        data['trans'] = (data['trans'] - min_trans) / (max_trans - min_trans)
        data['trans'] =  data['trans'].to(dtype=torch.float32)
        data['trans'] =  data['trans'].reshape(3)

        # 4. genders预处理 转为uint8
        # 布尔类型的存储通常以字节为单位，
        # 因此在存储大量布尔值时，每个布尔值都会占用一个字节，而不是单独的位。
        data['gender'] = torch.tensor(data['gender'], dtype = torch.uint8).unsqueeze(0)
        
        # 5. shape, pose的转格式
        data['shape'] = (data['shape'] - min_shape) / (max_shape - min_shape)
        data['pose'] = (data['pose'] - min_pose) / (max_pose - min_pose)
        data['shape'] = data['shape'].to(dtype=torch.float32)
        data['pose'] = data['pose'].to(dtype=torch.float32)
        
        return data
def calculate_dataset_statistics(dataset):
    loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    
    # 初始化变量以保存统计信息
    min_shape = float('inf')
    max_shape = float('-inf')
    min_pose = float('inf')
    max_pose = float('-inf')
    min_trans = float('inf')
    max_trans = float('-inf')

    for data in loader:
        # 获取一个批次的数据
        shape = data['shape']
        pose = data['pose']
        trans = data['trans']

        # 更新统计信息
        min_shape = min(min_shape, shape.min())
        max_shape = max(max_shape, shape.max())
        min_pose = min(min_pose, pose.min())
        max_pose = max(max_pose, pose.max())
        min_trans = min(min_trans, trans.min())
        max_trans = max(max_trans, trans.max())

    return min_shape, max_shape, min_pose, max_pose, min_trans, max_trans

# 使用示例

if __name__ == "__main__":
    # 使用示例
    lmdb_path = '/root/pose_master/dataset/train_lmdb_gt'  # 替换为LMDB数据库的路径
    lmdb_dataset = SkeletonDatasetLMDB(lmdb_path,transform=True)
    # lmdb_dataset = SkeletonDatasetLMDB(lmdb_path,transform=None)
    # 获取数据集的长度
    batch_size = 8
    print("Dataset length:", len(lmdb_dataset))
    # train_loader = torch.utils.data.DataLoader(lmdb_dataset,
    #                                             batch_size=batch_size, 
    #                                            shuffle=True,
    #                                            num_workers = 0, 
    #                                            pin_memory=True)
    
    # for i,data in enumerate(train_loader):
    #     print(data.keys())
    #     print(data['image'].shape)
    #     break
    
    
    # min_shape, max_shape, min_pose, max_pose, min_trans, max_trans = calculate_dataset_statistics(lmdb_dataset)
    # print("Min/Max Shape:", min_shape, max_shape)
    # print("Min/Max Pose:", min_pose, max_pose)
    # print("Min/Max Trans:", min_trans, max_trans)

    '''
    1. test_lmdb:
        Min/Max Shape: tensor(-3.0000) tensor(2.9999)
        Min/Max Pose: tensor(-2.6839) tensor(4.1719)
        Min/Max Trans: tensor(-0.0241) tensor(1.5819)
    2. train_lmdb:
        Min/Max Shape: tensor(-3.0000, dtype=torch.float64) tensor(3.0000, dtype=torch.float64)
        Min/Max Pose: tensor(-2.8452, dtype=torch.float64) tensor(4.1845, dtype=torch.float64)
        Min/Max Trans: tensor(-0.0224, dtype=torch.float64) tensor(1.5980, dtype=torch.float64)
    -------
        Min/Max Shape: -3~3
        Min/Max Pose: -2.8452~4.1845
        Min/Max Trans: -0.0241~1.5980
    '''
    

    
    # for index in range(0, len(lmdb_dataset)):
    #     data_item = lmdb_dataset.__getitem__(index)
    #     # print("Loaded data:", data_item.keys())
    #     print(index,end='\r')
    #     # print(data_item['image'])
    #     contains_negative = torch.any(data_item['image'] > 200)

    #     if contains_negative:
    #         print("The tensor contains negative values.")
    #         flag = 1
    #         break
    #     else:

    #         pass

      # 加载数据
    # index = 0  # 选择要加载的数据项的索引
    # data_item = lmdb_dataset[index]
    # for key, value in data_item.items():

    #     try:
    #         shape = value.shape
    #         dtype = value.dtype
    #         print(f"Key: {key}, Shape: {shape}, Dtype: {dtype}")
    #     except:
    #         print(f"Key: {key}, {type(value)}")
    #         pass
    