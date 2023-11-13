import h5py
import cv2
import torch
import numpy as np
import os
import lmdb
import sys
import pickle


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
        data['trans'] =  data['trans'].to(dtype=torch.float32)
        data['trans'] =  data['trans'].reshape(3)

        # 4. genders预处理 转为uint8
        # 布尔类型的存储通常以字节为单位，
        # 因此在存储大量布尔值时，每个布尔值都会占用一个字节，而不是单独的位。
        data['gender'] = torch.tensor(data['gender'], dtype = torch.uint8).unsqueeze(0)
        
        # 5. shape, pose的转格式
        data['shape'] = data['shape'].to(dtype=torch.float32)
        data['pose'] = data['pose'].to(dtype=torch.float32)

        return data

if __name__ == "__main__":
    # 使用示例
    lmdb_path = r'D:\workspace\python_ws\pose-master\dataset\test_lmdb_gt'  # 替换为LMDB数据库的路径
    lmdb_dataset = SkeletonDatasetLMDB(lmdb_path,transform=True)
    # lmdb_dataset = SkeletonDatasetLMDB(lmdb_path,transform=None)
    # 获取数据集的长度
    batch_size = 8
    print("Dataset length:", len(lmdb_dataset))
    train_loader = torch.utils.data.DataLoader(lmdb_dataset,
                                                batch_size=batch_size, 
                                               shuffle=True,
                                               num_workers = 0, 
                                               pin_memory=True)
    
    for i,data in enumerate(train_loader):
        print(data.keys())
        # print(data['trans'])
        # print((data['skeleton'].reshape(-1,24,3))[:,0])
        # print(data['trans'] - (data['skeleton'].reshape(-1,24,3))[:,0])
        skeleton_data = data['skeleton'].reshape(-1, 24, 3)
        # 从整个张量中减去第二个维度上的第一列
        skeleton = skeleton_data - skeleton_data[:, 0:1, :]

        # 现在，result 的形状是 (batch_size, 24, 3)，你可以继续进行后续计算
        print(skeleton)
        # skeletons = data['skeleton'].reshape(-1,24,3)
        # skeletons = skeletons - skeletons[:,0]
        # print(skeletons)
        break
    
    
  


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