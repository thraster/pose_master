import torch
import torch.nn as nn
import torch.optim as optim
import time
# torch.set_default_tensor_type(torch.DoubleTensor)

import sys

# 将模块所在的目录添加到模块搜索路径
module_location = 'D:\workspace\python_ws\pose-master'  # 将此路径替换为实际的模块所在目录
sys.path.append(module_location)

# from models.posenet_res import posenet
from models.posenet_res_smpl_embeded import posenet

import datetime
from torch.utils.tensorboard import SummaryWriter
from testing import test_model_smpl


def train(train_loader, test_loader, num_epochs=10, model = posenet, checkpoint_path = None):
    '''
    训练模型,默认选择resnet作为被训练的模型
    train_loader:训练集的dataloader实例
    test_loader:测试集的dataloader实例
    num_epochs:训练的epoch数
    checkpoint_path:不为空时从checkpoint文件加载模型权重,在其基础上继续训练
    '''
    test_loss_list = []
    # 初始化模型

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model(device = device).to(device)  # 根据实际情况调整模型初始化方式

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
    criterion = nn.MSELoss() # MSELoss = (1/n) * Σ(yᵢ - ȳ)²
    optimizer = optim.Adam(model.parameters(), lr=0.001)
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
        running_loss = 0.0
        time_batchs = 0.0
        fwd_batchs = 0.0
        loss_batchs = 0.0
        bwd_batchs = 0.0
        smpl_batchs = 0.0

        # 1.训练
        for i, data in enumerate(train_loader, 0):
            start_time = time.time()
            # data.keys() = ['skeleton', 'image', 'gender', 'trans']

            skeletons, images, genders, transs = data.values()

            # 将输入和标签移动到GPU上
            images = images.to(device)
            skeletons = skeletons.to(device)
            genders = genders.to(device)
            transs = transs.to(device)

            # 梯度清零
            optimizer.zero_grad()
            fwd_time1 = time.time()
            # 前向传播
            outputs, smpl_time = model(images, genders, transs)
            fwd_time2 = time.time()
            fwd_time = fwd_time2-fwd_time1

            # print(outputs.dtype)
            # print(outputs.shape)
            # print(skeletons.dtype)
            # print(skeletons.shape)


            # 默认使用float32数据类型进行训练
            loss_time1 = time.time()
            loss = criterion(outputs, skeletons)
            loss_time2 = time.time()
            loss_time = loss_time2-loss_time1

            # print(loss)
            # print(loss.shape)

            # 反向传播和优化

            # break
            bwd_time1 = time.time()
            loss.backward()
            bwd_time2 = time.time()
            bwd_time = bwd_time2-bwd_time1

            optimizer.step()

            # 打印统计信息
            running_loss += loss.item()
            
            if i*batch_size % 1024 == 0:  # 每训练过1024条数据打印一次, 判断是否保存, 集成停止功能
                
                print(f"[epoch {epoch}, iter {i}/{(train_data_batchs)}] loss: {running_loss / 1024 * batch_size}, time: {time_batchs},avg_time: {time_batchs/16}")
                print(f"forward time: {fwd_batchs}, loss time: {loss_batchs}, backward time: {bwd_batchs}, smpl_forward_time: {smpl_batchs}")
                running_loss = 0.0
                time_batchs = 0.0
                fwd_batchs = 0.0
                loss_batchs = 0.0
                bwd_batchs = 0.0
                smpl_batchs = 0.0

            # 添加标量 loss while training
            writer.add_scalar(tag=f"loss(training) MSEloss/batchs (batch_size ={batch_size})", scalar_value=loss,
                global_step=epoch * train_data_batchs + i)
            end_time = time.time()
            elapsed_time = end_time - start_time
            time_batchs += elapsed_time

            fwd_batchs += fwd_time
            loss_batchs += loss_time
            bwd_batchs += bwd_time
            smpl_batchs += smpl_time

            # print(f"一个batch运行时间：{elapsed_time} 秒")
        # 2.测试

        test_loss = test_model_smpl(model=model,epoch=epoch,test_loader=test_loader,device=device,criterion=criterion)
        
        # 添加标量 loss when test
        writer.add_scalar(tag="loss(testing) MSEloss/epochs", scalar_value=loss,
                global_step=epoch)
        test_loss_list.append(test_loss)

        # 3.保存checkpoint
        checkpoint = {
        'model_state_dict': model.state_dict(),
        # 'optimizer_state_dict': optimizer.state_dict(),
        'epochs': total_epoch, # 保存当前训练的 epoch 数
        'loss_funtion' : 'MSELoss',
        'optimizer' : 'Adam',
        'loss' : test_loss,
        'net' : model.name,
        # 可以保存其他超参数信息
        }
        torch.save(checkpoint, f'checkpoints/last_{model.name}.pth')
        print(f"last checkpoint saved! test loss = {test_loss}")
        
        # 每50个epoch储存一下？
        if epoch%50 == 0:
            torch.save(checkpoint, f'checkpoints/{model.name}_epoch{epoch}.pth')
        
        if epoch == 0:
            min_loss = test_loss # 记录初值作为min_loss

        elif test_loss < min_loss:
            # flag = 0
            min_loss = test_loss
            torch.save(checkpoint, f'checkpoints/best_{model.name}.pth')
            print(f"best checkpoint saved! = {min_loss}")

        # 如果连续5个epoch test loss没有再下降，停止训练
        # elif test_loss >= min_loss:
            # flag += 1
        
        # elif flag >= 5:
        #     break

    writer.close()
    print("训练完成")
    for i,los in enumerate(test_loss_list):
        print(f"epoch {i}, test loss = {los}")

# 用法示例
if __name__ == "__main__":

    # from load_dataset import SkeletonDataset
    # from load_dataset_hdf5 import SkeletonDatasetHDF5
    from load_dataset_mat import SkeletonDatasetMAT

    # import time

    # wait_time = 7200  # 等待1小时，可以根据需要设置等待时间

    # print("开始等待...")
    # time.sleep(wait_time)  # 暂停程序执行，等待指定的时间
    # print("等待结束，继续执行...")

    # train_data = SkeletonDataset(r'dataset\train',True)
    # test_data = SkeletonDataset(r'dataset\test',True)

    train_data = SkeletonDatasetMAT(r'F:\pose_master_dataset_mat\train')
    test_data = SkeletonDatasetMAT(r'F:\pose_master_dataset_mat\test')

    # train_data = SkeletonDataset(r'F:\pose_master_dataset\train')
    # test_data = SkeletonDataset(r'F:\pose_master_dataset\test')


    # for i in range(dataset.__len__()):
    #     sample = dataset[i]

    #     print(i, sample['skeleton'], sample['image'])
    #     if i >= 4:
    #         break

    batch_size = 64 # =64时,forward一次1.8s

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True,num_workers = 4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True,num_workers = 4, pin_memory=True)

    train(train_loader = test_loader, test_loader = test_loader, num_epochs=200, model=posenet, checkpoint_path=r'checkpoints\best_resnet+smpl(embeded).pth')
