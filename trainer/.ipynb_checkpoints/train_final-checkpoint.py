import torch
import torch.nn as nn
import torch.optim as optim
import time
# torch.set_default_tensor_type(torch.DoubleTensor)
import argparse
import sys

# 将模块所在的目录添加到模块搜索路径
module_location = '/root/pose_master'  # 将此路径替换为实际的模块所在目录
sys.path.append(module_location)

# from models.posenet_res import posenet
from models.with_mlp import mobile_with_mlp

import datetime
from torch.utils.tensorboard import SummaryWriter
from testing import test_model_final

# define the.
min_shape = -3
max_shape = 3
min_pose = -2.8452
max_pose = 4.1845
min_trans = -0.0241
max_trans = 1.5980


def train(train_loader, test_loader, num_epochs=10, model = mobile_with_mlp, checkpoint_path = None):
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

    optimizer_backbone = optim.Adam(model.backbone.parameters(), lr=0.00005, weight_decay=0.005)
    optimizer_mlp_m = optim.Adam(model.mlp_m.parameters(), lr=0.0002, weight_decay=0.005)
    optimizer_mlp_f = optim.Adam(model.mlp_f.parameters(), lr=0.0002, weight_decay=0.005)
    criterion = nn.L1Loss()
    
    train_data_batchs = len(train_loader)
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

    print(f"训练数据量: {train_data_batchs * batch_size}条, 分为{train_data_batchs}个batchs")
    print(f"batch size = {batch_size}")

    # 初始化tensorboard
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    # writer = SummaryWriter(log_dir=f'log/{model.name}/exp{current_date}')
    writer = {
        'loss(train)': SummaryWriter(f"/root/tf-logs/{model.name}/exp{current_date}"), #必须要不同的writer
        'loss(test)': SummaryWriter(f"/root/tf-logs/{model.name}/exp{current_date}"),

    }
    stop_flag = 0
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
            
            # transs = data['trans'].to(device)
            # shapes = data['shape'].to(device)
            # poses = data['pose'].to(device)
            # 梯度清零
            optimizer_backbone.zero_grad()
            optimizer_mlp_m.zero_grad()
            optimizer_mlp_f.zero_grad()

            # 前向传播
            x, _, joints = model(images, genders)
            loss = criterion(joints, skeletons)

            # 反向传播和优化

            # break
            loss.backward()
            optimizer_backbone.step()
            optimizer_mlp_m.step()
            optimizer_mlp_f.step()
            # break
            # 打印统计信息
            running_loss += loss.item()
            
            if i % 16 == 0:  # 每16个batch打印一次, 判断是否保存, 集成停止功能
                
                print(f"[epoch {epoch}, iter {i}/{(train_data_batchs)}] loss: {running_loss / 16 * batch_size:.6f}, time: {time_batchs:.6f}s ,avg_time: {time_batchs/batch_size*1000:.6f}ms")
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
        print(f"一个epoch运行时间：{time_epoch:.6f} 秒")
        time_epoch_list.append(time_epoch)

        test_loss = test_model_final(model=model,test_loader=test_loader,device=device,criterion=criterion)
        
        # 添加标量 loss when test
        writer['loss(test)'].add_scalar(tag="loss(testing) MSEloss/epochs", scalar_value=loss,
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
        if epoch == 0:
            min_loss = test_loss # 记录初值作为min_loss
       
        torch.save(checkpoint, f'/root/pose_master/my_checkpoints/exp2/last_{model.name}.pth')
        print(f"last checkpoint saved! test loss = {test_loss}")
        print(f"min loss = {min_loss}")
        
        # 每50个epoch储存一下？
        if epoch%50 == 0:
            torch.save(checkpoint, f'/root/pose_master/my_checkpoints/exp2/{model.name}_epoch{epoch}.pth')
        
        
        if test_loss > min_loss:
            stop_flag += 1
            if stop_flag >= 10:
                break
                
        elif test_loss <= min_loss:
            stop_flag = 0
            min_loss = test_loss
            torch.save(checkpoint, f'/root/pose_master/my_checkpoints/exp2/best_{model.name}.pth')
            print(f"best checkpoint saved! = {min_loss}")

    writer['loss(train)'].close()
    writer['loss(test)'].close()
    print("训练完成")
    for i,los in enumerate(test_loss_list):
        print(f"epoch {i}, test loss = {los}")
    total_time = 0.0
    for i,time_epoch in enumerate(time_epoch_list):
        total_time += time_epoch

    print(f"total time = {total_time}")
    print(f"avg time = {total_time/len(time_epoch_list)}")
    


def parse_args():
    parser = argparse.ArgumentParser(description='Your script description')

    # 添加命令行参数
    parser.add_argument('--train_data_path', type=str, default='pose_master/dataset/train_lmdb_gt/', help='Path to the training LMDB dataset')
    parser.add_argument('--test_data_path', type=str, default='pose_master/dataset/test_imdb_gt/', help='Path to the testing LMDB dataset')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training and testing')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to save the model checkpoints')

    args = parser.parse_args()
    return args

# 用法示例
if __name__ == "__main__":
    from load_dataset_lmdb_mod1 import SkeletonDatasetLMDB
 

    args = parse_args()
    test_data = SkeletonDatasetLMDB('/root/pose_master/dataset/test_imdb_gt/',  transform = True)
    train_data = SkeletonDatasetLMDB('/root/pose_master/dataset/train_lmdb_gt/', transform = True)

    batch_size = args.batch_size

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True,num_workers = 0, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True,num_workers = 0, pin_memory=True)


    train(train_loader = train_loader , test_loader = test_loader, num_epochs=100, model=mobile_with_mlp,checkpoint_path = '/root/pose_master/my_checkpoints/exp2/last_mobienetv2_with_mlp_refine.pth')
