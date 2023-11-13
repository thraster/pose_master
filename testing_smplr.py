import torch
import torch.nn as nn
import load_dataset
import time

def test_model_gender_split(model_f, model_m, test_loader, device, criterion):
    '''
    test函数,用于一个epoch结束后的test,并保存该epoch的checkpoint

    model:被测试的模型
    epoch:epoch数
    test_laoder:测试集的dataloader实例
    device:运行的设备
    criterion:损失函数
    
    '''
    
    # 将模型设置为评估模式
    model_f.eval()
    model_m.eval()

    print('testing...')
    # 定义损失函数（根据您的任务选择合适的损失函数）
    # criterion = nn.MSELoss()  # 示例中使用了二分类交叉熵损失

    test_loss_shape = 0.0
    test_loss_pose = 0.0
    lenth = len(test_loader)
    
    with torch.no_grad():
        
        for i, data in enumerate(test_loader):
            skeletons, images, genders = data.values()

            # 将输入和标签移动到GPU上（如果可用）
            skeleton_data = data['skeleton'].reshape(-1,24,3)
            skeletons = (skeleton_data - skeleton_data[:, 0:1, :]).to(device)
            genders = data['gender'].to(device)
            shape_gt = data['shape'].to(device)
            pose_gt = data['pose'].to(device)

            
            # 根据gender的取值将x分成x_f和x_m
            skeletons_f = skeletons[genders.squeeze() == 0]  # 选择gender为0的行作为女性数据
            skeletons_m = skeletons[genders.squeeze() == 1]  # 选择gender为1的行作为男性数据

            shape_gt_f = shape_gt[genders.squeeze() == 0] 
            shape_gt_m = shape_gt[genders.squeeze() == 1] 

            pose_gt_f = pose_gt[genders.squeeze() == 0]  # 选择gender为0的行作为女性数据
            pose_gt_m = pose_gt[genders.squeeze() == 1]  # 选择gender为1的行作为男性数据



            shape_f, pose_f = model_f(skeletons_f)
            shape_m, pose_m = model_m(skeletons_m)

            shape_loss_f = criterion(shape_f, shape_gt_f)
            pose_loss_f = criterion(pose_f, pose_gt_f)
            
            shape_loss_m = criterion(shape_m, shape_gt_m)
            pose_loss_m = criterion(pose_m, pose_gt_m)
            
            # 累积损失
            test_loss_shape += shape_loss_f.item()+shape_loss_m.item()
            test_loss_pose += pose_loss_f.item()+pose_loss_m.item()

            if i % 100 == 99:
                print(f"current progress: {i/lenth*100:.2f}%")
            # # 计算准确率
            # predicted = (outputs > 0.5).float()  # 二分类阈值通常为0.5
            # total += skeletons.size(0)
            # correct += (predicted == skeletons).sum().item()
        # test_loss = 0
    # 输出测试结果
    print(f"1. Test shape loss: {test_loss_shape / lenth}")
    print(f"2. Test pose loss: {test_loss_pose / lenth}")
    return test_loss_shape / lenth, test_loss_pose / lenth


def test_model(model, test_loader, device, criterion):
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

    test_loss_shape = 0.0
    test_loss_pose = 0.0
    lenth = len(test_loader)
    
    with torch.no_grad():
        
        for i, data in enumerate(test_loader):


            # 将输入和标签移动到GPU上（如果可用）
            skeleton_data = data['skeleton'].reshape(-1,24,3)
            skeletons = (skeleton_data - skeleton_data[:, 0:1, :]).reshape(-1,72).to(device)
            # genders = data['gender'].to(torch.float32)
            shape_gt = data['shape'].to(device)
            pose_gt = data['pose'].to(device)


            # inputs = torch.cat((skeletons,genders), dim=1).to(device) #gender连接在第一位

            # 根据gender的取值将x分成x_f和x_m
 

            shape, pose = model(skeletons)

            shape_loss = criterion(shape, shape_gt)
            pose_loss = criterion(pose, pose_gt)

            # 累积损失
            test_loss_shape += shape_loss.item()
            test_loss_pose += pose_loss.item()

            if i % 100 == 99:
                print(f"current progress: {i/lenth*100:.2f}%")
            # # 计算准确率
            # predicted = (outputs > 0.5).float()  # 二分类阈值通常为0.5
            # total += skeletons.size(0)
            # correct += (predicted == skeletons).sum().item()
        # test_loss = 0
    # 输出测试结果
    print(f"1. Test shape loss: {test_loss_shape / lenth}")
    print(f"2. Test pose loss: {test_loss_pose / lenth}")
    return test_loss_shape / lenth, test_loss_pose / lenth




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