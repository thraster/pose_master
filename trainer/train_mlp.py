import torch
import torch.nn as nn
import torch.optim as optim
import time
# torch.set_default_tensor_type(torch.DoubleTensor)

import sys

# 将模块所在的目录添加到模块搜索路径
module_location = 'D:\workspace\python_ws\pose-master-pack'  # 将此路径替换为实际的模块所在目录
sys.path.append(module_location)

# from models.posenet_res import posenet
from models.smplr import SMLPR_MLP
from testing_smplr import test_model
import datetime
from torch.utils.tensorboard import SummaryWriter



def train(train_loader, test_loader,model = SMLPR_MLP, num_epochs=10,  checkpoint_path = None):
    '''
    训练模型,默认选择resnet作为被训练的模型
    train_loader:训练集的dataloader实例
    test_loader:测试集的dataloader实例
    num_epochs:训练的epoch数
    checkpoint_path:不为空时从checkpoint文件加载模型权重,在其基础上继续训练
    '''
    test_loss_shape_list = []
    test_loss_pose_list = []
    time_epoch_list = []
    # 初始化模型

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    model = model(device)


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
    criterion = nn.L1Loss() # MSELoss = (1/n) * Σ(yᵢ - ȳ)²
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    train_data_batchs = len(train_loader)

    print(f"训练数据量: {train_data_batchs * batch_size}条, 分为{train_data_batchs}个batchs")
    print(f"batch size = {batch_size}")

    # 初始化tensorboard
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    writer = SummaryWriter(log_dir=f'log/{model.name}/exp{current_date}')

    for epoch in range(total_epoch, num_epochs):
        
        print("training...")
        model.train()

        print(f"current epoch: {epoch}")
        running_loss_shape = 0.0
        running_loss_pose = 0.0
        time_batchs = 0.0
        time_epoch = 0.0
        torch.autograd.set_detect_anomaly(True)

        # 1.训练
        for i, data in enumerate(train_loader):
            
            start_time = time.time()
            # data.keys() = ['skeleton', 'image', 'gender', 'trans']

            skeleton_data = data['skeleton'].reshape(-1,24,3)
            skeletons = (skeleton_data - skeleton_data[:, 0:1, :]).reshape(-1,72).to(device)
            images = data['image']
            # genders = data['gender'].to(torch.float32)
            shape_gt = data['shape'].to(device)
            pose_gt = data['pose'].to(device)


            # inputs = torch.cat((skeletons,genders), dim=1).to(device) #gender连接在第一位


            shape, pose= model(skeletons)



            optimizer.zero_grad()
            
            # 默认使用float32数据类型进行训练
            shape_loss = criterion(shape, shape_gt)
            pose_loss = criterion(pose, pose_gt)

            # 反向传播和优化

            # break
            shape_loss.backward()
            pose_loss.backward()
            optimizer.step()


            # break
            # 打印统计信息
            running_loss_shape += shape_loss.item()
            running_loss_pose += pose_loss.item()

            if i*batch_size % 1024 == 0:  # 每训练过1024条数据打印一次, 判断是否保存, 集成停止功能
                
                print(f"[epoch {epoch}, iter {i}/{(train_data_batchs)}] shape loss: {running_loss_shape / 1024 * batch_size}, pose loss: {running_loss_pose / 1024 * batch_size}, time: {time_batchs},avg_time: {time_batchs/16}")
                time_batchs = 0.0
                # 添加标量 loss while training
                writer.add_scalar(tag=f"shape loss(training) L1loss/batchs (batch_size ={batch_size})", scalar_value=running_loss_shape,
                    global_step=epoch * train_data_batchs + i)
                writer.add_scalar(tag=f"pose loss(training) L1loss/batchs (batch_size ={batch_size})", scalar_value=running_loss_pose,
                    global_step=epoch * train_data_batchs + i)
                running_loss_pose = 0.0
                running_loss_shape = 0.0

            end_time = time.time()
            elapsed_time = end_time - start_time
            time_batchs += elapsed_time
            time_epoch += elapsed_time

        # 2.测试
        print(f"一个epoch运行时间：{time_epoch} 秒")
        time_epoch_list.append(time_epoch)

        test_loss_shape, test_loss_pose = test_model(model=model,test_loader=test_loader,device=device,criterion=criterion)
        
        # 添加标量 loss when test
        writer.add_scalar(tag="shape loss(testing) L1loss/epochs", scalar_value=test_loss_shape,
                global_step=epoch)
        test_loss_shape_list.append(test_loss_shape)

        writer.add_scalar(tag="pose loss(testing) L1loss/epochs", scalar_value=test_loss_pose,
                global_step=epoch)
        test_loss_pose_list.append(test_loss_pose)

        # 3.保存checkpoint
        checkpoint = {
        'model_state_dict': model.state_dict(),
        # 'optimizer_state_dict': optimizer.state_dict(),
        'epochs': epoch+1, # 保存当前训练的 epoch 数
        'loss_funtion' : 'MSELoss',
        'optimizer' : 'Adam',
        'shape loss' : test_loss_shape,
        'pose loss' : test_loss_pose,
        'net' : model.name,
        # 可以保存其他超参数信息
        }
        torch.save(checkpoint, f'checkpoints/last_{model.name}_f.pth')
        print(f"last checkpoint saved!")
        print(f"1. Test shape loss: {test_loss_shape}")
        print(f"2. Test pose loss: {test_loss_pose}")
   
        
        # 每50个epoch储存一下？
        if epoch%50 == 0:
            torch.save(checkpoint, f'checkpoints/f{model.name}_epoch{epoch}_f.pth')
        
        if epoch == 0:
            min_loss = test_loss_shape + test_loss_pose # 记录初值作为min_loss

        elif test_loss_shape + test_loss_pose < min_loss:
            # flag = 0
            min_loss = test_loss_shape + test_loss_pose
            torch.save(checkpoint, f'checkpoints/best_{model.name}_f.pth')
            print(f"best checkpoint saved!")
            print(f"1. Test shape loss: {test_loss_shape}")
            print(f"2. Test pose loss: {test_loss_pose}")


    writer.close()
    print("训练完成")
    for i,shape_l, pose_l in enumerate(zip(test_loss_shape, test_loss_pose)):
        print(f"epoch {i}, shape loss = {shape_l}, pose loss = {pose_l}")
    total_time = 0.0
    for i,time_epoch in enumerate(time_epoch_list):
        total_time += time_epoch

    print(f"totaltime = {total_time}")
    print(f"avgtime = {total_time/len(time_epoch_list)}")

# 用法示例
if __name__ == "__main__":
    from load_dataset_lmdb import SkeletonDatasetLMDB
 
    torch.autograd.set_detect_anomaly(True)
    train_data = SkeletonDatasetLMDB(r'D:\workspace\python_ws\pose-master\dataset\train_lmdb_gt_f', transform = True)
    test_data = SkeletonDatasetLMDB(r'D:\workspace\python_ws\pose-master\dataset\test_lmdb_gt_f',  transform = True)


    batch_size = 64

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True,num_workers = 0, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True,num_workers = 0, pin_memory=True)

    train(train_loader = train_loader , test_loader = test_loader, num_epochs=101, model=SMLPR_MLP,checkpoint_path = None)
    # train(train_loader = train_loader , test_loader = test_loader, num_epochs=101, model=mobilenet,checkpoint_path = None)
