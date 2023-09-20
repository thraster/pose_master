import torch
import torch.nn as nn
import torch.optim as optim
from vgg import VGG16  # 从 vgg 模块中导入 VGG16 模型
import read
from torch.utils.data import DataLoader, TensorDataset


def train_vgg16(train_loader, num_epochs=10):
    # 初始化模型
    model = VGG16()  # 根据实际情况调整模型初始化方式

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # 训练循环
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data

            # 梯度清零
            optimizer.zero_grad()

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            # 打印统计信息
            running_loss += loss.item()
            if i % 100 == 99:  # 每 100 个小批次打印一次
                print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}")
                running_loss = 0.0

    print("训练完成")









# 用法示例
if __name__ == "__main__":
    import os
    import cv2  # 用于读取图像文件
    import scipy.io  # 用于读取.mat文件

    # # 定义文件夹路径
    # folder_path = r'dataset\test\test_roll0_f_lay_set14_1500'

    # # 初始化 X 和 Y 列表
    # X = []  # 用于存储图像数据
    # y = []  # 用于存储.mat文件数据

    # # 遍历文件夹中的文件
    # for filename in os.listdir(folder_path):
    #     file_path = os.path.join(folder_path, filename)
        
    #     # 检查文件类型并处理
    #     if filename.endswith('.png'):
    #         # 读取图像文件并将其添加到 X 列表
    #         image = read.load_png(file_path)
    #         X.append(image)
    #     elif filename.endswith('.mat'):
    #         # 读取.mat文件并将其添加到 Y 列表
    #         mat_data = read.load_mat
    #         y.append(mat_data)
    #     else:
    #         # 可以根据需要处理其他类型的文件
    #         pass

    # # 现在，X 列表包含了所有图像数据，Y 列表包含了所有.mat文件数据

        
    #     # # 假设您有图像数据 (X) 和对应的标签 (y)，这里假设它们都是 PyTorch Tensor
    #     # X = torch.randn(100, 3, 224, 224)  # 100 个图像，每个图像的大小为 3x224x224
    #     # y = torch.randint(0, 10, (100,))  # 100 个标签，假设有 10 个类别

    # # 创建一个数据集对象
    # X = torch.stack([torch.tensor(image) for image in X])
    # y = [torch.tensor(data) for data in y]
    # 定义文件夹路径
    folder_path = r'dataset\test\test_roll0_f_lay_set14_1500'

    # 初始化 X 和 Y 列表
    X = []  # 用于存储图像数据
    Y = []  # 用于存储.mat文件数据

    # 遍历文件夹中的文件
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # 检查文件类型并处理
        if filename.endswith('.png'):
            # 读取图像文件并将其添加到 X 列表
            image = cv2.imread(file_path)
            if image is not None:
                X.append(image)
        elif filename.endswith('.mat'):
            # 读取.mat文件并将其添加到 Y 列表
            mat_data = scipy.io.loadmat(file_path)
            Y.append(mat_data)
        else:
            # 可以根据需要处理其他类型的文件
            pass
    dataset = TensorDataset(X, Y)

    # 创建一个数据加载器
    batch_size = 32  # 每个小批次的大小
    shuffle = True  # 是否随机打乱数据
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    train_vgg16(data_loader, num_epochs=10)
