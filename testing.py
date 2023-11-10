import torch
import torch.nn as nn
import load_dataset
import time

def test_model(model, epoch, test_loader, device, criterion):
    '''
    test函数,用于一个epoch结束后的test,并保存该epoch的checkpoint

    model:被测试的模型
    epoch:epoch数
    test_laoder:测试集的dataloader实例
    device:运行的设备
    criterion:损失函数
    
    '''
    
    # 将模型设置为评估模式
    model.eval()
    print('testing...')
    # 定义损失函数（根据您的任务选择合适的损失函数）
    # criterion = nn.MSELoss()  # 示例中使用了二分类交叉熵损失

    test_loss = 0.0
    lenth = len(test_loader)
    
    with torch.no_grad():
        
        for i, data in enumerate(test_loader):
            skeletons, images, genders = data.values()

            # 将输入和标签移动到GPU上（如果可用）
            images = images.to(device)
            skeletons = skeletons.to(device)

            # 前向传播
            outputs = model(images)
            
            # 计算损失
            loss = criterion(outputs, skeletons)
            
            # 累积损失
            test_loss += loss.item()
            if i % 100 == 99:
                print(f"current progress: {i/lenth*100:.2f}%")
            # # 计算准确率
            # predicted = (outputs > 0.5).float()  # 二分类阈值通常为0.5
            # total += skeletons.size(0)
            # correct += (predicted == skeletons).sum().item()
        # test_loss = 0
    # 输出测试结果
    print(f"Test Loss: {test_loss / lenth}")

    # checkpoint = {
    #             'model_state_dict': model.state_dict(),
    #             # 'optimizer_state_dict': optimizer.state_dict(),
    #             'epochs': epoch, # 保存当前训练的 epoch 数
    #             'loss_funtion' : 'MSELoss',
    #             'optimizer' : 'Adam',
    #             'test_loss' : test_loss / lenth,
    #             'net' : "vgg16",
    #             # 可以保存其他超参数信息
    #             }
    # torch.save(checkpoint, f'checkpoints/checkpoint_epoch{epoch}.pth')
    # print(f"epoch {epoch} checkpoint saved!")

    return test_loss / lenth
    # accuracy = 100 * correct / total
    # print(f"Test Accuracy: {accuracy:.2f}%")

def test_model_smpl(model, test_loader, device, criterion):
    '''
    test函数,用于一个epoch结束后的test,并保存该epoch的checkpoint

    model:被测试的模型
    epoch:epoch数
    test_laoder:测试集的dataloader实例
    device:运行的设备
    criterion:损失函数
    
    '''
    
    # 将模型设置为评估模式
    model.eval()
    print('testing...')
    # 定义损失函数（根据您的任务选择合适的损失函数）
    # criterion = nn.MSELoss()  # 示例中使用了二分类交叉熵损失

    test_loss = 0.0
    lenth = len(test_loader)
    
    with torch.no_grad():
        
        for i, data in enumerate(test_loader):
        
            # lenth = data['gender'].size(0)
            skeletons = data['skeleton'].to(device)
            images = data['image'].to(device)
            genders = data['gender'].to(device)
            transs = data['trans'].to(device)

            # 前向传播
            _, outputs = model(images, genders)

            loss = criterion(outputs, skeletons)
            # print(loss)

            # end_time = time.time()

            # print(f"求loss代码执行时间：{end_time-forward_time} 秒")
            # 累积损失
            test_loss += loss.item()
            if i % 100 == 99:
                print(f"current progress: {i/lenth*100:.2f}%")
            # # 计算准确率
            # predicted = (outputs > 0.5).float()  # 二分类阈值通常为0.5
            # total += skeletons.size(0)
            # correct += (predicted == skeletons).sum().item()
        # test_loss = 0
    # 输出测试结果
    print(f"Test Loss: {test_loss / lenth}")



    return test_loss / lenth
    # accuracy = 100 * correct / total
    # print(f"Test Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    model = vgg.VGG16()
    checkpoint = torch.load(r'checkpoints\last.pth')
    # print(f"loading checkpoint [{checkpoint_path}] successed!")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    test_data = load_dataset.SkeletonDataset(r'dataset\test',True)

    test_loader = torch.utils.data.DataLoader(test_data, batch_size = 8, shuffle=False)

    

    test_model(model=model,epoch=0, test_loader=test_loader,device=device,criterion=nn.MSELoss())