import torch
import torch.nn as nn
import torch.optim as optim
from vgg import VGG16  # 从 vgg 模块中导入 VGG16 模型
import math
from torch.utils.data import DataLoader, TensorDataset


def train_vgg16(train_loader, num_epochs=10):
    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VGG16().to(device)  # 根据实际情况调整模型初始化方式
    print('model initialized on', device)

    # 定义损失函数和优化器
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(num_epochs):
        print(f"current epoch: {epoch}")
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # data.keys() = ['skeleton', 'image']
            labels, inputs = data.values()

            # 将输入和标签移动到GPU上
            inputs = inputs.to(device).float() 
            labels = labels.to(device)

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

            if math.isnan(running_loss):
                # 处理 NaN 损失的操作，例如终止训练或采取其他修复方法
                print("NaN损失发生，终止训练或采取修复措施")
                print("inputs: ",inputs)
                print("labels: ",labels)
                print("outputs :",outputs)
                # 保存模型checkpoint
                checkpoint = {
                'model_state_dict': model.state_dict(),
                # 'optimizer_state_dict': optimizer.state_dict(),
                'inputs': inputs,
                'labels': labels,
                'epoch': num_epochs,  # 保存当前训练的 epoch 数
                # 可以保存其他超参数信息
                }
                torch.save(checkpoint, f'checkpoints/error_checkpoint_{i}.pth')
                break  # 终止训练


            if i % 100 == 99:  # 每 100 个小批次打印一次
                print(f"[epoch {epoch}, iter {i}] loss: {running_loss / 100:.3f}")
                running_loss = 0.0

                # # 保存模型checkpoint
                # checkpoint = {
                # 'model_state_dict': model.state_dict(),
                # # 'optimizer_state_dict': optimizer.state_dict(),
                # 'epoch': num_epochs,  # 保存当前训练的 epoch 数
                # # 可以保存其他超参数信息
                # }
                # torch.save(checkpoint, f'checkpoints/checkpoint_{i}.pth')


    print("训练完成")


# 用法示例
if __name__ == "__main__":

    from load_dataset import SkeletonDataset
    
    dataset = SkeletonDataset(r'dataset\train',True)
    
    # for i in range(dataset.__len__()):
    #     sample = dataset[i]

    #     print(i, sample['skeleton'], sample['image'])
    #     if i >= 4:
    #         break

    batch_size = 8
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    train_vgg16(data_loader, num_epochs=10)
