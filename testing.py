import torch
import torch.nn as nn
# import load_dataset
import time
import torch.nn.functional as F
min_shape = -3
max_shape = 3
min_pose = -2.8452
max_pose = 4.1845
min_trans = -0.0241
max_trans = 1.5980

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
            _, _, outputs = model(images)
            
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
    print(f"Test Loss: {test_loss / lenth:.6f}")

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
def test_model_smpl_exp1(model, test_loader, device, criterion, stage):
    '''
    test函数,用于一个epoch结束后的test,并保存该epoch的checkpoint

    model:被测试的模型
    epoch:epoch数
    test_laoder:测试集的dataloader实例
    device:运行的设备
    criterion:损失函数
    
    '''
    
    # 将模型设置为评估模式
    if stage == 1:
        criterion = nn.L1Loss()
    elif stage == 2:
        criterion = nn.MSELoss() # MSELoss = (1/n) * Σ(yᵢ - ȳ)²
    model.eval()
    print(f'testing... {model.name}')
    # 定义损失函数（根据您的任务选择合适的损失函数）
    # criterion = nn.MSELoss()  # 示例中使用了二分类交叉熵损失

    test_loss = 0.0
    lenth = len(test_loader)
    
    with torch.no_grad():
        
        for i, data in enumerate(test_loader):
        
            # lenth = data['gender'].size(0)
            images = data['image'].to(device)
            skeletons = data['skeleton'].to(device)
            genders = data['gender'].to(device)
            
            transs = data['trans'].to(device)
            shapes = data['shape'].to(device)
            poses = data['pose'].to(device)

            # 前向传播
            x, _, joints = model(images, genders)

            if stage == 1:
                pred_shapes = (x[:,0:10] - min_shape) / (max_shape - min_shape)
                pred_poses = (x[:,10:82] - min_pose) / (max_pose - min_pose)
                pred_trans = (x[:,82:85] - min_trans) / (max_trans - min_trans)
                
                loss = criterion(pred_shapes, shapes)+criterion(pred_poses, poses)+criterion(pred_trans, transs)
                lossname = 'L1loss of shape, pose, trans(normalized)'
            elif stage == 2:
                loss = criterion(joints, skeletons)
                lossname = 'L2loss of joints'
            test_loss += loss.item()
            if i % 100 == 99:
                print(f"current progress: {i/lenth*100:.2f}%")
            # # 计算准确率
            # predicted = (outputs > 0.5).float()  # 二分类阈值通常为0.5
            # total += skeletons.size(0)
            # correct += (predicted == skeletons).sum().item()
        # test_loss = 0
    # 输出测试结果
    print(f"Test Loss {lossname}: {test_loss / lenth}")

    return test_loss / lenth
    # accuracy = 100 * correct / total
    # print(f"Test Accuracy: {accuracy:.2f}%")

def test_model_final(model, test_loader, device, criterion):
    '''
    test函数,用于一个epoch结束后的test,并保存该epoch的checkpoint

    model:被测试的模型
    epoch:epoch数
    test_laoder:测试集的dataloader实例
    device:运行的设备
    criterion:损失函数
    
    '''
    
    # 将模型设置为评估模式
    criterion = nn.L1Loss()

    model.eval()
    print(f'testing... {model.name}')
    # 定义损失函数（根据您的任务选择合适的损失函数）
    # criterion = nn.MSELoss()  # 示例中使用了二分类交叉熵损失

    test_loss = 0.0
    lenth = len(test_loader)
    
    with torch.no_grad():
        
        for i, data in enumerate(test_loader):
        
            # lenth = data['gender'].size(0)
            images = data['image'].to(device)
            skeletons = data['skeleton'].to(device)
            genders = data['gender'].to(device)
            # 前向传播
            _, _, joints = model(images, genders)
            loss = criterion(joints, skeletons)
            test_loss += loss.item()
            if i % 100 == 99:
                print(f"current progress: {i/lenth*100:.2f}%")
            # # 计算准确率
            # predicted = (outputs > 0.5).float()  # 二分类阈值通常为0.5
            # total += skeletons.size(0)
            # correct += (predicted == skeletons).sum().item()
        # test_loss = 0
    # 输出测试结果
    print(f"Test Loss (L1loss of joints): {test_loss / lenth}")

    return test_loss / lenth
    
def test_model_refine(model, test_loader, device, criterion):
    '''
    test函数,用于一个epoch结束后的test,并保存该epoch的checkpoint

    model:被测试的模型
    epoch:epoch数
    test_laoder:测试集的dataloader实例
    device:运行的设备
    criterion:损失函数
    
    '''
    
    # 将模型设置为评估模式
    criterion = nn.L1Loss()

    model.eval()
    print(f'testing... {model.name}')
    # 定义损失函数（根据您的任务选择合适的损失函数）
    # criterion = nn.MSELoss()  # 示例中使用了二分类交叉熵损失

    test_loss = 0.0
    lenth = len(test_loader)
    
    with torch.no_grad():
        
        for i, data in enumerate(test_loader):
        
            # lenth = data['gender'].size(0)
            images = data['image'].to(device)
            # skeletons = data['skeleton'].to(device)
            # genders = data['gender'].to(device)
            transs = data['trans'].to(device)
            shapes = data['shape'].to(device)
            poses = data['pose'].to(device)
            # 前向传播
            feature_maps = model(images)

            losses = []
            for loss_idx in range(0,len(feature_maps) // 3):
                pred_shapes = F.adaptive_avg_pool2d(feature_maps[loss_idx*3], (1, 1)).squeeze()
                pred_poses = F.adaptive_avg_pool2d(feature_maps[loss_idx*3 + 1], (1, 1)).squeeze()
                pred_transs = F.adaptive_avg_pool2d(feature_maps[loss_idx*3 + 2], (1, 1)).squeeze()

                pred_shapes = (pred_shapes - min_shape) / (max_shape - min_shape)
                pred_poses = (pred_poses - min_pose) / (max_pose - min_pose)
                pred_transs = (pred_transs - min_trans) / (max_trans - min_trans)

                # print(pred_shapes.shape, pred_poses.shape, pred_transs.shape)

                losses.append(criterion(pred_shapes, shapes))
                losses.append(criterion(pred_poses, poses))
                losses.append(criterion(pred_transs, transs))
            

            loss = losses[0]
            for loss_idx in range(1, len(losses)):
                loss += losses[loss_idx]

            test_loss += loss.item()
            if i % 100 == 99:
                print(f"current progress: {i/lenth*100:.2f}%")
            # # 计算准确率
            # predicted = (outputs > 0.5).float()  # 二分类阈值通常为0.5
            # total += skeletons.size(0)
            # correct += (predicted == skeletons).sum().item()
        # test_loss = 0
    # 输出测试结果
    print(f"Test Loss (L1loss of joints): {test_loss / lenth}")

    return test_loss / lenth

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