import torch
import torch.nn as nn
import torch.optim as optim
from resnet import resnet18  # 从 vgg 模块中导入 VGG16 模型
import math
from torch.utils.data import DataLoader, TensorDataset
from test import test_model


def train_resnet(train_loader, test_loader, num_epochs=10, checkpoint_path = None):
    '''
    训练模型,默认选择VGG16作为被训练的模型
    train_loader:训练集的dataloader实例
    test_loader:测试集的dataloader实例
    num_epochs:训练的epoch数
    checkpoint_path:不为空时从checkpoint文件加载模型权重,在其基础上继续训练
    '''
    test_loss_list = []
    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = resnet18().to(device)  # 根据实际情况调整模型初始化方式

    if checkpoint_path != None:
        checkpoint = torch.load(checkpoint_path)
        total_epoch = checkpoint['epochs']
        print(f"loading checkpoint [{checkpoint_path}] successed!")
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"loading state_dict successed!")
        print(f"checkpoint info: epoch = {checkpoint['epochs']}")
    else:
        total_epoch = 0


    print('model initialized on', device)
    # is_nan = False
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    train_data_len = len(train_data) // batch_size
    for epoch in range(num_epochs):
        print("training...")

        model.train()
        print(f"current epoch: {epoch}")
        running_loss = 0.0

        # 1.训练
        for i, data in enumerate(train_loader, 0):
            # data.keys() = ['skeleton', 'image']

            labels, inputs = data.values()

            # 将输入和标签移动到GPU上
            inputs = inputs.to(device)
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

            if i % 1000 == 999:  # 每 1000 个小批次打印一次, 判断是否保存, 集成停止功能
                # # 初始化min_loss
                # if i == 99 and epoch == 0:
                #     min_loss = running_loss

                # # 保存模型checkpoint
                # checkpoint = {
                # 'model_state_dict': model.state_dict(),
                # # 'optimizer_state_dict': optimizer.state_dict(),
                # 'iters': i,
                # 'epochs': total_epoch + epoch, # 保存当前训练的 epoch 数
                # 'loss_funtion' : 'MSELoss',
                # 'optimizer' : 'Adam',
                # 'loss' : loss.item(),
                # 'net' : "vgg16",
                # # 可以保存其他超参数信息
                # }
                # torch.save(checkpoint, f'checkpoints/last.pth')
                # print(f"last checkpoint saved! 100 iter avg loss = {running_loss / 100:.3f}")
                

                # if running_loss <= min_loss:
                #     min_loss = running_loss
                #     torch.save(checkpoint, f'checkpoints/best.pth')
                #     print(f"best checkpoint saved! 100 iter avg loss = {min_loss / 100:.3f}")


                print(f"[epoch {epoch}, iter {i}/{train_data_len/batch_size}] loss: {running_loss / 100:.3f}")
                running_loss = 0.0

        # 2.测试
        test_loss = test_model(model=model,epoch=epoch,test_loader=test_loader,device=device,criterion=criterion)
        test_loss_list.append(test_loss)
        # 3.保存checkpoint

        checkpoint = {
        'model_state_dict': model.state_dict(),
        # 'optimizer_state_dict': optimizer.state_dict(),
        'epochs': total_epoch + epoch, # 保存当前训练的 epoch 数
        'loss_funtion' : 'MSELoss',
        'optimizer' : 'Adam',
        'loss' : test_loss,
        'net' : "vgg16",
        # 可以保存其他超参数信息
        }
        torch.save(checkpoint, f'checkpoints/resnet_last.pth')
        print(f"last checkpoint saved! test loss = {test_loss}")

        if epoch == 0:
            min_loss = test_loss # 记录初值作为min_loss

        elif test_loss < min_loss:
            flag = 0
            min_loss = test_loss
            torch.save(checkpoint, f'checkpoints/resnet_best.pth')
            print(f"best checkpoint saved! = {min_loss}")

        # 如果连续5个epoch test loss没有再下降，停止训练
        elif test_loss >= min_loss:
            flag += 1
        
        elif flag >= 5:
            break

    print("训练完成")
    for i,los in enumerate(test_loss_list):
        print(f"epoch {i}, test loss = {los}")

# 用法示例
if __name__ == "__main__":

    from load_dataset import SkeletonDataset
    
    train_data = SkeletonDataset(r'dataset\train',True)
    test_data = SkeletonDataset(r'dataset\test',True)
    # for i in range(dataset.__len__()):
    #     sample = dataset[i]

    #     print(i, sample['skeleton'], sample['image'])
    #     if i >= 4:
    #         break

    batch_size = 8
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

    train_resnet(train_loader = test_loader, test_loader = test_loader, num_epochs=200)

