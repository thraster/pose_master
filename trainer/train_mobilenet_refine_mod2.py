
import torch
import torch.nn as nn
import torch.optim as optim
import time
# torch.set_default_tensor_type(torch.DoubleTensor)
import argparse
import sys
import torch.nn.functional as F
# 将模块所在的目录添加到模块搜索路径
module_location = '/root/pose_master'  # 将此路径替换为实际的模块所在目录
sys.path.append(module_location)

# from models.posenet_res import posenet
from models.mobilenet_refine import mobilenet_refine
from models.mobilenet_refine_beta import mobilenet_refine_beta

import datetime
from torch.utils.tensorboard import SummaryWriter
from testing_refine import test_model_refine
from smpl.smpl_torch_batch import SMPLModel

# define the.
min_shape = -3
max_shape = 3
min_pose = -2.8452
max_pose = 4.1845
min_trans = -0.0241
max_trans = 1.5980


def get_parameters(model, predicate):
    for module in model.modules():
        for param_name, param in module.named_parameters():
            if predicate(module, param_name):
                yield param


def get_parameters_conv(model, name):
    return get_parameters(model, lambda m, p: isinstance(m, nn.Conv2d) and m.groups == 1 and p == name)


def get_parameters_conv_depthwise(model, name):
    return get_parameters(model, lambda m, p: isinstance(m, nn.Conv2d)
                                              and m.groups == m.in_channels
                                              and m.in_channels == m.out_channels
                                              and p == name)


def get_parameters_bn(model, name):
    return get_parameters(model, lambda m, p: isinstance(m, nn.BatchNorm2d) and p == name)


def train(train_loader, test_loader, num_epochs=100, model = mobilenet_refine, checkpoint_path = None, gender = 'm'):
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

    model = model(device).to(device)  # 根据实际情况调整模型初始化方式
 
    model.name = 'Mobilnet_refine_mod2(joint_supervized)'
    if gender == 'f':
        model_path_f = '/root/pose_master/smpl/basicModel_f_lbs_10_207_0_v1.0.0.pkl'
        smpl_forward = SMPLModel(device=device,model_path=model_path_f).to(device)
    elif gender == 'm':
        model_path_m = '/root/pose_master/smpl/basicmodel_m_lbs_10_207_0_v1.0.0.pkl'
        smpl_forward = SMPLModel(device=device,model_path=model_path_m).to(device)
    
    
    
    criterion = nn.L1Loss()
    
    train_data_batchs = len(train_loader)
    if checkpoint_path != None:
        checkpoint = torch.load(checkpoint_path)
        total_epoch = checkpoint['epochs']
        min_MPJPE = checkpoint['MPJPE']
        min_V2V = checkpoint['V2V']
        print(f"loading checkpoint successed!")
        print(f"loss of checkpoint will be taken as min_loss!")

        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"loading state_dict successed!")
        print(f"checkpoint info: epoch = {checkpoint['epochs']}")
    else:
        total_epoch = 0

    batch_size = args.batch_size
    print(f"训练数据量: {train_data_batchs * batch_size}条, 分为{train_data_batchs}个batchs")
    print(f"batch size = {batch_size}")

    # 初始化tensorboard
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    # writer = SummaryWriter(log_dir=f'log/{model.name}/exp{current_date}')
    writer = {
        'loss(train)': SummaryWriter(f"/root/tf-logs/{model.name}[{gender}]/exp{current_date}"), #必须要不同的writer
        'MPJPE': SummaryWriter(f"/root/tf-logs/{model.name}[{gender}]/exp{current_date}"),
        'V2V': SummaryWriter(f"/root/tf-logs/{model.name}[{gender}]/exp{current_date}"),
        'shape[1]': SummaryWriter(f"/root/tf-logs/{model.name}[{gender}]/exp{current_date}"),
        'pose[1]': SummaryWriter(f"/root/tf-logs/{model.name}[{gender}]/exp{current_date}"),
        'trans[1]': SummaryWriter(f"/root/tf-logs/{model.name}[{gender}]/exp{current_date}"),
        'shape[0]': SummaryWriter(f"/root/tf-logs/{model.name}[{gender}]/exp{current_date}"),
        'pose[0]': SummaryWriter(f"/root/tf-logs/{model.name}[{gender}]/exp{current_date}"),
        'trans[0]': SummaryWriter(f"/root/tf-logs/{model.name}[{gender}]/exp{current_date}"),
    }

    base_lr = 4e-5

    optimizer = optim.Adam([
        {'params': get_parameters_conv(model.model, 'weight')},
        {'params': get_parameters_conv_depthwise(model.model, 'weight'), 'weight_decay': 0},
        {'params': get_parameters_bn(model.model, 'weight'), 'weight_decay': 0},
        {'params': get_parameters_bn(model.model, 'bias'), 'lr': base_lr * 2, 'weight_decay': 0},
        {'params': get_parameters_conv(model.cpm, 'weight'), 'lr': base_lr},
        {'params': get_parameters_conv(model.cpm, 'bias'), 'lr': base_lr * 2, 'weight_decay': 0},
        {'params': get_parameters_conv_depthwise(model.cpm, 'weight'), 'weight_decay': 0},
        {'params': get_parameters_conv(model.initial_stage, 'weight'), 'lr': base_lr},
        {'params': get_parameters_conv(model.initial_stage, 'bias'), 'lr': base_lr * 2, 'weight_decay': 0},
        {'params': get_parameters_conv(model.refinement_stages, 'weight'), 'lr': base_lr * 4},
        {'params': get_parameters_conv(model.refinement_stages, 'bias'), 'lr': base_lr * 8, 'weight_decay': 0},
        {'params': get_parameters_bn(model.refinement_stages, 'weight'), 'weight_decay': 0},
        {'params': get_parameters_bn(model.refinement_stages, 'bias'), 'lr': base_lr * 2, 'weight_decay': 0},
    ], lr=base_lr, weight_decay=5e-4)
    
    stop_flag = 0
    for epoch in range(total_epoch, num_epochs):
        
        print(f"training... [{model.name}_{gender}]")

        model.train()
        print(f"current epoch: {epoch}")
        running_loss = 0.0
        time_batchs = 0.0
        time_epoch = 0.0

        # 1.训练
        for i, data in enumerate(train_loader):
            # break
            start_time = time.time()
            # data.keys() = ['skeleton', 'image', 'gender', 'trans']

            images = data['image'].to(device)
            skeletons = data['skeleton'].to(device)
            # genders = data['gender'].to(device)
            
            transs = data['trans'].to(device)
            shapes = data['shape'].to(device)
            poses = data['pose'].to(device)

            # 梯度清零
            optimizer.zero_grad()

            # 前向传播
            feature_maps = model(images)

            losses = []
            for loss_idx in range(0,len(feature_maps) // 3):
                pred_shapes = F.adaptive_avg_pool2d(feature_maps[loss_idx*3], (1, 1)).squeeze()
                pred_poses = F.adaptive_avg_pool2d(feature_maps[loss_idx*3 + 1], (1, 1)).squeeze()
                pred_transs = F.adaptive_avg_pool2d(feature_maps[loss_idx*3 + 2], (1, 1)).squeeze()
                pred_params = torch.cat((pred_shapes, pred_poses, pred_transs), dim = 1)
                
                
                _, pred_joints = smpl_forward(pose = pred_poses,
                                            betas = pred_shapes,
                                            trans= pred_transs
                                             )

                # print(pred_shapes.shape, pred_poses.shape, pred_transs.shape)
                # print(pred_joints.shape)

                losses.append(criterion(pred_joints.reshape(-1,72), skeletons))

            loss = losses[0]
            for loss_idx in range(1, len(losses)):
                loss += losses[loss_idx]
            # break
            loss.backward()
            # break
            optimizer.step()
            # break
            # 打印统计信息
            running_loss += loss.item()
            
            if i % 16 == 0:  # 每16个batch打印一次, 判断是否保存, 集成停止功能
                
                print(f"[epoch {epoch}, iter {i}/{(train_data_batchs)}] loss: {running_loss / 16 * batch_size:.6f}, time: {time_batchs:.6f}s ,avg_time: {time_batchs/batch_size*1000:.6f}ms")
                # print(f"forward time: {fwd_batchs}, loss time: {loss_batchs}, backward time: {bwd_batchs}, smpl_forward_time: {smpl_batchs}")
                running_loss = 0.0
                time_batchs = 0.0


            # 添加标量 loss while training
            writer['loss(train)'].add_scalar(tag=f"loss(training) L1Loss/batchs (batch_size ={batch_size})", scalar_value=loss,
                global_step=epoch * train_data_batchs + i)
            end_time = time.time()
            elapsed_time = end_time - start_time
            time_batchs += elapsed_time
            time_epoch += elapsed_time

            
        # 2.测试
        print(f"一个epoch运行时间：{time_epoch:.6f} 秒")
        time_epoch_list.append(time_epoch)

        MPJPE, V2V, shape1, shape0, pose1, pose0, trans1, trans0 = test_model_refine(model=model,test_loader=test_loader,device=device, gender = gender)
        
        # break
        # 添加标量 loss when test
        writer['MPJPE'].add_scalar(tag="MPJPE(test) /mm", scalar_value=MPJPE,
                global_step=epoch)
        test_loss_list.append(MPJPE)
        writer['V2V'].add_scalar(tag="V2V(test) /mm", scalar_value=V2V,
                global_step=epoch)
        test_loss_list.append(V2V)
        writer['shape[1]'].add_scalar(tag="shape loss [1]", scalar_value=shape1,
                global_step=epoch)
        writer['shape[0]'].add_scalar(tag="shape loss [0]", scalar_value=shape0,
                global_step=epoch)
        writer['pose[1]'].add_scalar(tag="pose loss [1]", scalar_value=pose1,
                global_step=epoch)
        writer['pose[0]'].add_scalar(tag="pose loss [0]", scalar_value=pose0,
                global_step=epoch)
        writer['trans[1]'].add_scalar(tag="trans loss [1]", scalar_value=trans1,
                global_step=epoch)
        writer['trans[0]'].add_scalar(tag="trans loss [0]", scalar_value=trans0,
                global_step=epoch)

        # 3.保存checkpoint
        checkpoint = {
        'model_state_dict': model.state_dict(),
        # 'optimizer_state_dict': optimizer.state_dict(),
        'epochs': epoch+1, # 保存当前训练的 epoch 数
        'loss_funtion' : 'MSELoss',
        'optimizer' : 'Adam',
        'MPJPE' : MPJPE,
        'V2V' : V2V,
        'net' : model.name,
        # 可以保存其他超参数信息
        }
        if epoch == 0:
            min_MPJPE = MPJPE
            min_V2V = V2V
       
        torch.save(checkpoint, f'/root/pose_master/my_checkpoints/refine/last_{model.name}_{gender}.pth')
        print(f"last checkpoint saved!")
        print(f"[min MPJPE and cooresponding V2V = {min_MPJPE}, {min_V2V}]")
        
        # 每50个epoch储存一下？
        if epoch%50 == 0:
            torch.save(checkpoint, f'/root/pose_master/my_checkpoints/refine/{model.name}_{gender}_epoch{epoch}.pth')
        
        
        if MPJPE > min_MPJPE:
            stop_flag += 1
            if stop_flag >= 50:
                break
                
        else:
            stop_flag = 0
            min_MPJPE = MPJPE
            min_V2V = V2V
            torch.save(checkpoint, f'/root/pose_master/my_checkpoints/refine/best_{model.name}_{gender}.pth')
            print(f"best checkpoint saved! MPJPE = {MPJPE}, V2V = {V2V}")
    
    # 关闭所有writer
    for _, subwriter in writer.items():
        subwriter.close()

    print("训练完成")
    total_time = 0.0
    for i,time_epoch in enumerate(time_epoch_list):
        total_time += time_epoch

    print(f"total time = {total_time}")
    print(f"avg time = {total_time / len(time_epoch_list)}")
    


def parse_args():
    parser = argparse.ArgumentParser(description='Your script description')

    # 添加命令行参数
    parser.add_argument('--gender', type=str, default='m', help='train model for given gender')
    parser.add_argument('--train_data_path', type=str, default='/root/pose_master/dataset/train_lmdb_gt_', help='Path to the training LMDB dataset')
    parser.add_argument('--test_data_path', type=str, default='/root/pose_master/dataset/test_lmdb_gt_', help='Path to the testing LMDB dataset')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training and testing')
    parser.add_argument('--num_epochs', type=int, default=300, help='Number of training epochs')
    parser.add_argument('--checkpoint_path', type=str, default='/root/pose_master/my_checkpoints/refine/last_Mobilnet_refine_mod2(joint_supervized)_', help='Path to loading model checkpoints')

    args = parser.parse_args()
    return args

# 用法示例
if __name__ == "__main__":
    # 使用load_dataset_lmdb, 在testing部分需要用到未作调整的shapes, transs, poses
    from load_dataset_lmdb import SkeletonDatasetLMDB
 

    args = parse_args()
    args_dict = vars(args)
    for arg_key, arg_value in args_dict.items():
        print(f"{arg_key}: {arg_value}")
    train_data = SkeletonDatasetLMDB(f'{args.train_data_path+args.gender}/',  transform = True)
    test_data = SkeletonDatasetLMDB(f'{args.test_data_path+args.gender}/', transform = True)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True,num_workers = 0, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=True,num_workers = 0, pin_memory=True)
    
    train(train_loader = train_loader , test_loader = test_loader, num_epochs=args.num_epochs, model=mobilenet_refine, checkpoint_path = f'{args.checkpoint_path+args.gender}.pth', gender = args.gender)



#     args = parse_args()
#     test_data = SkeletonDatasetLMDB('/root/pose_master/dataset/test_lmdb_gt_f/',  transform = True)
#     train_data = SkeletonDatasetLMDB('/root/pose_master/dataset/train_lmdb_gt_f/', transform = True)

#     batch_size = args.batch_size

#     train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True,num_workers = 0, pin_memory=True)
#     test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True,num_workers = 0, pin_memory=True)
#     train(train_loader = train_loader , test_loader = test_loader, num_epochs=300, model=mobilenet_refine, checkpoint_path = None, gender = 'f')
