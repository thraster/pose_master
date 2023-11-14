import torch
import torch.nn as nn
import torch.optim as optim
import time
# torch.set_default_tensor_type(torch.DoubleTensor)

import sys

# 将模块所在的目录添加到模块搜索路径
module_location = '/root/pose_master'  # 将此路径替换为实际的模块所在目录
sys.path.append(module_location)

from models.mobilenet_pretrained import pretrained_mobilenet


import datetime
from torch.utils.tensorboard import SummaryWriter
from testing import test_model_smpl


def train(train_loader, test_loader, num_epochs=10, model = pretrained_mobilenet, checkpoint_path = None):
    '''
    训练模型,默认选择resnet作为被训练的模型
    train_loader:训练集的dataloader实例
    test_loader:测试集的dataloader实例
    num_epochs:训练的epoch数
    checkpoint_path:不为空时从checkpoint文件加载模型权重,在其基础上继续训练
    '''
    test_loss_list = []
    time_epoch_list = []
    # 初始化模型

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model(device = device).to(device)  # 根据实际情况调整模型初始化方式
    model_path_f = r'E:\WorkSpace\python_ws\pose-master-pack\smpl\basicModel_f_lbs_10_207_0_v1.0.0.pkl'
    model_path_m = r'E:\WorkSpace\python_ws\pose-master-pack\smpl\basicmodel_m_lbs_10_207_0_v1.0.0.pkl'

    
    # 初始化male和female的smpl模型
    # smpl_m = SMPLModel(device=device,model_path=r'smpl\basicmodel_m_lbs_10_207_0_v1.0.0.pkl')
    # smpl_f = SMPLModel(device=device,model_path=r'smpl\basicModel_f_lbs_10_207_0_v1.0.0.pkl')

    if checkpoint_path != None:
        checkpoint = torch.load(checkpoint_path)
        total_epoch = checkpoint['epochs']
        min_loss = checkpoint['loss']
        print(f"loading checkpoint [{checkpoint_path}] successed!")
        print(f"loss of checkpoint [{checkpoint_path}] will be taken as min_loss!")

        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"loading state_dict successed!")
        print(f"checkpoint info: epoch = {checkpoint['epochs']}")
    else:
        total_epoch = 0


    print('model initialized on', device)
    # is_nan = False
    # 定义损失函数和优化器
    L2loss = nn.MSELoss() # MSELoss = (1/n) * Σ(yᵢ - ȳ)²
    L1loss = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    train_data_batchs = len(train_loader)

    print(f"训练数据量: {train_data_batchs * batch_size}条, 分为{train_data_batchs}个batchs")
    print(f"batch size = {batch_size}")

    # 初始化tensorboard
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    writer = {
        'loss(train)': SummaryWriter(f"tf-logs/mod2/{model.name}/exp{current_date}"), #必须要不同的writer
        'loss(test)': SummaryWriter(f"tf-logs/mod2/{model.name}/exp{current_date}"),

    }
    for epoch in range(total_epoch, num_epochs):
        
        print("training...")

        model.train()
        print(f"current epoch: {epoch}")
        running_loss = 0.0
        time_batchs = 0.0
        time_epoch = 0.0

        # 1.训练
        for i, data in enumerate(train_loader):
            
            start_time = time.time()
            # data.keys() = ['skeleton', 'image', 'gender', 'trans']

            images = data['image'].to(device)
            skeletons = data['skeleton'].to(device)
            genders = data['gender'].to(device)
            transs = data['trans'].to(device)
            
            shapes = data['shape'].to(device)
            poses = data['pose'].to(device)
            
            # break
            # print(images.shape)
            # 梯度清零
            optimizer.zero_grad()

            # 前向传播
            x, meshes, joints = model(images, genders)

            # meshes, joints = SMPLforward(x.clone(), genders)

            # 默认使用float32数据类型进行训练
            joints_loss = L2loss(joints, skeletons)
            shape_loss = L1loss(x[:, :10], shapes)
            pose_loss = L1loss(x[:, 10:82], poses)
            
            loss = shape_loss + pose_loss +joints_loss
            # 反向传播和优化

            # break
            loss.backward()
            optimizer.step()
            # break
            # 打印统计信息
            running_loss += loss.item()
            
            if i*batch_size % 1024 == 0:  # 每训练过1024条数据打印一次, 判断是否保存, 集成停止功能
                
                print(f"[epoch {epoch}, iter {i}/{(train_data_batchs)}] loss: {running_loss / 1024 * batch_size}, time: {time_batchs},avg_time: {time_batchs/16}")
                # print(f"forward time: {fwd_batchs}, loss time: {loss_batchs}, backward time: {bwd_batchs}, smpl_forward_time: {smpl_batchs}")
                running_loss = 0.0
                time_batchs = 0.0


            # 添加标量 loss while training
            writer['loss(train)'].add_scalar(tag=f"loss(training) MSEloss/batchs (batch_size ={batch_size})", scalar_value=loss,
            global_step=epoch * train_data_batchs + i)
            end_time = time.time()
            elapsed_time = end_time - start_time
            time_batchs += elapsed_time
            time_epoch += elapsed_time

            # print(f"一个batch运行时间：{elapsed_time} 秒")
        # 2.测试
        print(f"一个epoch运行时间：{time_epoch} 秒")
        time_epoch_list.append(time_epoch)

        test_loss = test_model_smpl(model=model,test_loader=test_loader,device=device,criterion=L2loss)
        
        # 添加标量 loss when test
        writer['loss(test)'].add_scalar(tag="loss(testing) MSEloss/epochs", scalar_value=test_loss,
                global_step=epoch)
        test_loss_list.append(test_loss)

        # 3.保存checkpoint
        checkpoint = {
        'model_state_dict': model.state_dict(),
        # 'optimizer_state_dict': optimizer.state_dict(),
        'epochs': epoch+1, # 保存当前训练的 epoch 数
        'loss_funtion' : 'MSELoss',
        'optimizer' : 'Adam',
        'loss' : test_loss,
        'net' : model.name,
        # 可以保存其他超参数信息
        }
        torch.save(checkpoint, f'/root/pose_master/my_checkpoints/mod2/last_{model.name}.pth')
        print(f"last checkpoint saved! test loss = {test_loss}")
        
        # 每50个epoch储存一下？
        if epoch%50 == 0:
            torch.save(checkpoint, f'/root/pose_master/my_checkpoints/mod2{model.name}_epoch{epoch}.pth')
        
        if epoch == 0:
            min_loss = test_loss # 记录初值作为min_loss

        elif test_loss < min_loss:
            # flag = 0
            min_loss = test_loss
            torch.save(checkpoint, f'/root/pose_master/my_checkpoints/mod2/best_{model.name}.pth')
            print(f"best checkpoint saved! = {min_loss}")


    writer['loss(train)'].close()
    writer['loss(test)'].close()
    print("训练完成")
    for i,los in enumerate(test_loss_list):
        print(f"epoch {i}, test loss = {los}")
    total_time = 0.0
    for i,time_epoch in enumerate(time_epoch_list):
        total_time += time_epoch

    print(f"totaltime = {total_time}")
    print(f"avgtime = {total_time/len(time_epoch_list)}")

# 用法示例
if __name__ == "__main__":
    from load_dataset_lmdb import SkeletonDatasetLMDB
 

    test_data = SkeletonDatasetLMDB('/root/pose_master/dataset/test_imdb_gt/',  transform = True)
    train_data = SkeletonDatasetLMDB('/root/pose_master/dataset/train_lmdb_gt/', transform = True)


    batch_size = 256

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True,num_workers = 0, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True,num_workers = 0, pin_memory=True)

    train(train_loader = train_loader , test_loader = test_loader, num_epochs=201, model=pretrained_mobilenet,checkpoint_path = '/root/pose_master/my_checkpoints/best_pretrained_mobilenetv2_smpl.pth')
