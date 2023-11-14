import torch
import torch.nn as nn
import torch.optim as optim
import time
import sys

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import MultipleLocator

from models.resnet18_pretrained import pretrained_resnet18
from models.resnet50_pretrained import pretrained_resnet50
from models.resnet50_pretrained import pretrained_resnet50
from models.mobilenet_pretrained import pretrained_mobilenet
from models.with_mlp import mobile_with_mlp
from smpl.smpl_torch_batch import SMPLModel


def calculate_mpjpe(joints_true, joints_pred):
    """
    计算 Mean Per-Joint Position Error (MPJPE)。

    参数:
        joints_true (Tensor): 真实关节点坐标，形状为 (batch_size, 24, 3)。
        joints_pred (Tensor): 模型预测的关节点坐标，形状为 (batch_size, 24, 3)。

    返回:
        mpjpe (Tensor): MPJPE 值，形状为 (batch_size, 24)。
    """
    joints_true = joints_true.reshape(-1,24,3)
    joints_pred = joints_pred.reshape(-1,24,3)
    # 计算欧几里德距离
    squared_error = torch.sum((joints_true - joints_pred) ** 2, dim=2)
    euclidean_distance = torch.sqrt(squared_error)

    # 计算每个关节点的平均误差
    mpjpe = torch.mean(euclidean_distance, dim=1)

    return mpjpe

def calculate_v2v_error(joints_true, joints_pred):
    """
    计算 Vertex-to-Vertex Error (V2V)。

    参数:
        joints_true (Tensor): 真实关节点坐标，形状为 (batch_size, 6980, 3)。
        joints_pred (Tensor): 模型预测的关节点坐标，形状为 (batch_size, 6980, 3)。

    返回:
        v2v_error (Tensor): V2V 错误值，形状为 (batch_size,).
    """
    # 计算欧几里德距离
    squared_error = torch.sum((joints_true - joints_pred) ** 2, dim=2)
    euclidean_distance = torch.sqrt(squared_error)

    # 对每个样本计算平均距离
    v2v_error = torch.mean(euclidean_distance, dim=1)

    return v2v_error


def eval_smpl_mesh(model, checkpoint_path, test_loader, device, isbatch=True):
    '''
    model: 被测试的模型
    checkpoint_path: 模型对应的权重文件
    test_laoder: 测试集的dataloader实例
    device: 运行的设备
    isbatch: 是测量一个batch还是在整个test上测量
    '''
    model = model(device = device).to(device)
    checkpoint = torch.load(checkpoint_path)
    print(f"loading checkpoint [{checkpoint_path}] successed!")
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"loading state_dict successed!")
    print(f"checkpoint info: epoch = {checkpoint['epochs']}")
    print('model initialized on', device)

    # 将模型设置为评估模式
    model.eval()
    print('testing...')
    # 定义损失函数（根据您的任务选择合适的损失函数）
    criterion = nn.MSELoss()  # 示例中使用了二分类交叉熵损失
    #初始化smpl, 用于结合ground truth pose和shape生成ground truth mesh
    model_path_f = 'pose_master/smpl/basicModel_f_lbs_10_207_0_v1.0.0.pkl'
    smpl_f = SMPLModel(device=device,model_path=model_path_f).to(device)

    model_path_m = 'pose_master/smpl/basicmodel_m_lbs_10_207_0_v1.0.0.pkl'
    smpl_m = SMPLModel(device=device,model_path=model_path_m).to(device)
    
    with torch.no_grad():

        MeanMeshLoss = 0.0
        for i, data in enumerate(test_loader):
            # print(data)
            print(data.keys())
            
            # labels, inputs, genders, trans = data.values()

            skeletons = data['skeleton'].to(device)
            images = data['image'].to(device)
            genders = data['gender'].to(device)
            transs = data['trans'].to(device)
            # transs = torch.stack([
            #                     data['trans'][:, 0] - 0.286,
            #                     data['trans'][:, 1] - 0.286 + 0.15,
            #                     0.12 - data['trans'][:, 2]
            #                 ], dim=1).to(device)
            # print(trans.shape)
            # break
            shapes = data['shape'].to(device)
            poses = data['pose'].to(device)

            lenth = images.size(0)
            shapes_f = shapes[genders.squeeze() == 0]  # 选择gender为0的行作为女性数据
            shapes_m = shapes[genders.squeeze() == 1]  # 选择gender为1的行作为男性数据

            poses_f = poses[genders.squeeze() == 0]  # 选择gender为0的行作为女性数据
            poses_m = poses[genders.squeeze() == 1]  # 选择gender为1的行作为男性数据

            transs_f = transs[genders.squeeze() == 0]  # 选择gender为0的行作为女性数据
            transs_m = transs[genders.squeeze() == 1]  # 选择gender为1的行作为男性数据

            mesh_f, joints_f = smpl_f(
                pose = poses_f ,
                betas = shapes_f,
                trans = transs_f
            )

            mesh_m, joints_m = smpl_m(
                pose = poses_m ,
                betas = shapes_m,
                trans = transs_m
            )

            batch_size = images.size(0)

            mesh_gt = torch.empty((batch_size, 6890, 3)).to(device)  # 创建与x相同大小的空张量
            mesh_gt[genders.squeeze() == 0] = mesh_f  # 将x_f数据放回原始位置
            mesh_gt[genders.squeeze() == 1] = mesh_m  # 将x_m数据放回原始位置

            joints_gt = torch.empty((batch_size, 24, 3)).to(device)  # 创建与x相同大小的空张量
            joints_gt[genders.squeeze() == 0] = joints_f  # 将x_f数据放回原始位置
            joints_gt[genders.squeeze() == 1] = joints_m  # 将x_m数据放回原始位置


            start_time = time.time()
            # 前向传播
            mesh, joints = model(images, genders)
            forward_time = time.time()

            # 计算损失
            
            batchMeanMeshLoss = criterion(mesh, mesh_gt)

            MeanMeshLoss += batchMeanMeshLoss.item()

            if isbatch == True:
                print(f"forward代码执行时间: {forward_time - start_time} 秒")
                print(f"平均一个iter执行时间: {(forward_time - start_time)/lenth}")
                break
            else:
                lenth = len(test_loader)
                print(f"evalutating... [{i}/{lenth}]")
                
        # 输出测试结果
        print(f"mean mesh loss = {MeanMeshLoss / lenth}")

    return MeanMeshLoss, mesh_gt.cpu(), mesh.cpu()   # gt, pred

def eval_smpl_joints(model, checkpoint_path, test_loader, device, isbatch=True):
    '''
    model: 被测试的模型
    checkpoint_path: 模型对应的权重文件
    test_laoder: 测试集的dataloader实例
    device: 运行的设备
    isbatch: 是测量一个batch还是在整个test上测量
    '''
    model = model(device = device).to(device)
    checkpoint = torch.load(checkpoint_path)
    # print(f"loading checkpoint [{checkpoint_path}] successed!")
    model.load_state_dict(checkpoint['model_state_dict'])
    # print(f"loading state_dict successed!")
    # print(f"checkpoint info: epoch = {checkpoint['epochs']}")
    # print('model initialized on', device)

    # 将模型设置为评估模式
    model.eval()
    print('evaluating...')
    # 定义损失函数（根据您的任务选择合适的损失函数）
    criterion = nn.MSELoss()  # 示例中使用了二分类交叉熵损失
    batch_num = len(test_loader)

    print(batch_num)

    total_time = 0.0
    
    with torch.no_grad():
        totalMPJPE = 0.0
        for i, data in enumerate(test_loader):
            # print(f"batch [{i}]",end = '\r')
            # print(data)
            # print(data.keys())
            if i == 0:
                batch_size = data['gender'].size(0)
                print(batch_size)

            # labels, inputs, genders, trans = data.values()
            skeletons = data['skeleton'].to(device)
            images = data['image'].to(device)
            genders = data['gender'].to(device)
            transs = data['trans'].to(device)
            # print(trans.shape)
            # break
           
            start_time = time.time()
            # 前向传播
            mesh, joints = model(images, genders)
            forward_time = time.time()

            # 计算损失
            # print(joints.shape)
            # print(skeletons.shape)

            batchMPJPE = criterion(joints*1000, skeletons*1000)

            totalMPJPE += batchMPJPE.item()

            total_time += forward_time-start_time

            if isbatch == True:
                
                print(f"forward代码执行时间: {total_time} 秒")
                print(f"平均一个iter执行时间: {(forward_time-start_time)/batch_size}")
                break
            else:
                batch_lenth = len(test_loader)
                print(f"evaluating... [{i}/{batch_num}]", end='\r')
                
        # 输出测试结果
        MeanJointsLoss = totalMPJPE / batch_size/ (i+1)
        print(f"mean joints Loss: {MeanJointsLoss}")
        print(f"forward代码执行时间: {total_time} 秒")
        print(f"平均一个iter执行时间: {(forward_time-start_time)/batch_size/ (i+1)}")
        # print(f"mean mesh loss = {MeanMeshLoss / lenth}")


    return MeanJointsLoss, joints.cpu(), skeletons.cpu(), mesh.cpu() # gt, pred


def visualize_joints(joints1,joints2):
    """
    Visualize 24x3 joint coordinates in a 3D plot.

    Args:
        joints (torch.Tensor): A tensor of shape (24, 3) containing 3D joint coordinates.
        joints1 - the predicted joints
        joints2 - the ground truth joints
    """
    connections = [(12, 15), (12, 14), (12, 13), 
                   (13, 16), (16, 18), (18, 20), (20, 22), 
                   (14, 17), (17, 19), (19, 21),(21, 23), 
                   (14, 9), (13, 9), (9, 6), (6, 3), (3, 0), (0, 2), (0, 1),
                   (1, 4), (4, 7), (7, 10),
                   (2, 5), (5, 8), (8, 11)]  # 示例连接信息

    # 创建一个3D图形
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')


    # 设置X坐标轴的单位刻度
    ax.xaxis.set_major_locator(MultipleLocator(0.2))  # 在这里设置单位刻度的值

    # 设置Y坐标轴的单位刻度
    ax.yaxis.set_major_locator(MultipleLocator(0.2))  # 在这里设置单位刻度的值

    # 设置Z坐标轴的单位刻度
    ax.zaxis.set_major_locator(MultipleLocator(0.2))  # 在这里设置单位刻度的值


    # 绘制第一组关节点，使用红色
    ax.scatter(joints1[:, 0], joints1[:, 1], joints1[:, 2], c='r', marker='o', label='Joints Set 1')

    # 绘制第二组关节点，使用蓝色
    ax.scatter(joints2[:, 0], joints2[:, 1], joints2[:, 2], c='b', marker='o', label='Joints Set 2')

    # 绘制关节连接线
    for connection in connections:
        joint1, joint2 = connection
        x1, y1, z1 = joints1[joint1]
        x2, y2, z2 = joints1[joint2]
        ax.plot([x1, x2], [y1, y2], [z1, z2], c='r')

        x1, y1, z1 = joints2[joint1]
        x2, y2, z2 = joints2[joint2]
        ax.plot([x1, x2], [y1, y2], [z1, z2], c='b')

    # 设置坐标轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # 显示图形

    # for i in range(len(joints)):
    #     x = joints[i, 0]
    #     y = joints[i, 1]
    #     z = joints[i, 2]
    #     ax.scatter(x, y, z, c='r', marker='o')
    #     ax.text(x, y, z, i)  # 添加关节点标号

    # 自定义图例信息
    labels = ['predict joints', 'ground truth joints']
    # 添加图例
    ax.legend(labels)

    # 设置三个轴比例尺一致
    ax.axis('equal')  

    plt.show()

def visualize_mesh(mesh1, mesh2):
    """
    Visualize two sets of (6980, 3) mesh vertices in a 3D plot.

    Args:
        mesh1 (torch.Tensor): A tensor of shape (6980, 3) containing the first set of mesh vertices.
        mesh2 (torch.Tensor): A tensor of shape (6980, 3) containing the second set of mesh vertices.
    """

    # 创建一个3D图形
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 设置X坐标轴的单位刻度
    ax.xaxis.set_major_locator(MultipleLocator(0.2))  # 在这里设置单位刻度的值

    # 设置Y坐标轴的单位刻度
    ax.yaxis.set_major_locator(MultipleLocator(0.2))  # 在这里设置单位刻度的值

    # 设置Z坐标轴的单位刻度
    ax.zaxis.set_major_locator(MultipleLocator(0.2))  # 在这里设置单位刻度的值

    # 绘制第一个mesh的顶点，使用红色
    ax.scatter(mesh1[:, 0], mesh1[:, 1], mesh1[:, 2], c='r', marker='o', label='Mesh 1')

    # 绘制第二个mesh的顶点，使用蓝色
    ax.scatter(mesh2[:, 0], mesh2[:, 1], mesh2[:, 2], c='b', marker='o', label='Mesh 2')

    # 设置坐标轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # 自定义图例信息
    labels = ['predict mesh', 'ground truth mesh']
    # 添加图例
    ax.legend(labels)

    # 使用set_box_aspect函数设置坐标轴的单位比例
    ax.axis('equal')  # 单位比例在这里设置为1，您可以根据需要调整

    # 显示图形
    plt.show()


def eval_smpl(model, checkpoint_path, test_loader, device, isbatch=True):
    '''
    model: 被测试的模型
    checkpoint_path: 模型对应的权重文件
    test_laoder: 测试集的dataloader实例
    device: 运行的设备
    isbatch: 是测量一个batch还是在整个test上测量
    ---------------------
    返回结果:
    joint_pred, mesh_pred, joint_gt, mesh_gt, MPJPE, V2V
    '''
    if model == pretrained_mobilenet:
        model = model(device = device, issmpl = True).to(device)
    else:
        model = model(device = device).to(device)
    checkpoint = torch.load(checkpoint_path)
    print(f"loading checkpoint [{checkpoint_path}] successed!")
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"epoch = {checkpoint['epochs']}")

    # 将模型设置为评估模式
    model.eval()
    print('evaluating...')
    batch_num = len(test_loader)

    print("total batchs: ",batch_num)
    model_path_f = '/root/pose_master/smpl/basicModel_f_lbs_10_207_0_v1.0.0.pkl'
    smpl_f = SMPLModel(device=device,model_path=model_path_f).to(device)

    model_path_m ='/root/pose_master/smpl/basicmodel_m_lbs_10_207_0_v1.0.0.pkl'
    smpl_m = SMPLModel(device=device,model_path=model_path_m).to(device)
    
    total_time = 0.0
    
    with torch.no_grad():
        totalMPJPE = 0.0
        totalV2V = 0.0
        for i, data in enumerate(test_loader):
            # print(f"batch [{i}]",end = '\r')
            # print(data)
            # print(data.keys())
            if i == 0:
                batch_size = data['gender'].size(0)
                print(batch_size)
            # for key, value in data.items():
            #     try:
            #         shape = value.shape
            #         dtype = value.dtype
            #         print(f"Key: {key}, Shape: {shape}, Dtype: {dtype}")
            #     except:
            #         print(f"Key: {key}, {type(value)}")
            #         pass
            # labels, inputs, genders, trans = data.values()
            skeletons = data['skeleton'].to(device)
            images = data['image'].to(device)
            genders = data['gender'].to(device)
            transs = data['trans'].to(device)
            # print(trans.shape)
            # break
           
            start_time = time.time()
            前向传播
            if model.name == 'pretrained_mobilenetv2_smpl' or 'mobilenetv2_with_mlp_refine':
                _, mesh, joints = model(images, genders)
            else:
                mesh, joints = model(images, genders)
            forward_time = time.time()

            # 计算ground truth
            shapes = data['shape'].to(device)
            poses = data['pose'].to(device)
            shapes_f = shapes[genders.squeeze() == 0]  # 选择gender为0的行作为女性数据
            shapes_m = shapes[genders.squeeze() == 1]  # 选择gender为1的行作为男性数据
            poses_f = poses[genders.squeeze() == 0]  # 选择gender为0的行作为女性数据
            poses_m = poses[genders.squeeze() == 1]  # 选择gender为1的行作为男性数据
            transs_f = transs[genders.squeeze() == 0]  # 选择gender为0的行作为女性数据
            transs_m = transs[genders.squeeze() == 1]  # 选择gender为1的行作为男性数据
            mesh_f, joints_f = smpl_f(
                pose = poses_f ,
                betas = shapes_f,
                trans = transs_f
            )
            mesh_m, joints_m = smpl_m(
                pose = poses_m ,
                betas = shapes_m,
                trans = transs_m
            )
            batch_size = images.size(0)
            mesh_gt = torch.empty((batch_size, 6890, 3)).to(device)  # 创建与x相同大小的空张量
            mesh_gt[genders.squeeze() == 0] = mesh_f  # 将x_f数据放回原始位置
            mesh_gt[genders.squeeze() == 1] = mesh_m  # 将x_m数据放回原始位置

            joints_gt = torch.empty((batch_size, 24, 3)).to(device)  # 创建与x相同大小的空张量
            joints_gt[genders.squeeze() == 0] = joints_f  # 将x_f数据放回原始位置
            joints_gt[genders.squeeze() == 1] = joints_m  # 将x_m数据放回原始位置

            # 计算损失
            batchMPJPE = calculate_mpjpe((joints_gt*1000).cpu(), (joints*1000).cpu())

            batchV2V = calculate_v2v_error((mesh_gt*1000).cpu(), (mesh*1000).cpu())


            totalMPJPE += torch.sum(batchMPJPE)
            totalV2V += torch.sum(batchV2V)

            total_time += forward_time-start_time

            if isbatch == True:
                
                # print(f"forward代码执行总时间: {total_time*1000} ms")
                # print(f"平均一个iter执行时间: {((forward_time-start_time)/batch_size)*1000} ms")
                break
            else:
                batch_lenth = len(test_loader)
                print(f"evaluating... [{i}/{batch_num}]", end='\r')
                
        # 输出测试结果
        MPJPE = totalMPJPE / batch_size/ (i+1)
        V2V = totalV2V / batch_size/ (i+1)
        print(f"Mean Per-Joint Position Error (MPJPE): {MPJPE}")
        print(f"vertex-to-vertex error (v2v): {V2V}")
        print(f"forward代码执行总时间: {total_time*1000} ms")
        print(f"平均一个iter执行时间: {((forward_time-start_time)/batch_size/ (i+1))*1000} ms")
        # print(f"mean mesh loss = {MeanMeshLoss / lenth}")


    return joints.cpu(), mesh.cpu(), skeletons.cpu(), mesh_gt.cpu(), MPJPE, V2V

# 用法示例
if __name__ == "__main__":
    from load_dataset_lmdb import SkeletonDatasetLMDB

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_data = SkeletonDatasetLMDB('/root/pose_master/dataset/test_imdb_gt',  transform = True)
    batch_size = 64
    # shuffle要为true才能正常跑
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True,num_workers = 0, pin_memory=True)

    # 1. joint 测试部分
    # MeanJointsLoss, joints, joints_groundtruth,meshs = eval_smpl_joints(model=posenet,
    #                                         checkpoint_path = r'D:\workspace\python_ws\pose-master\checkpoints\exp2_nov10\best_resnet18_smpl.pth',
    #                                         test_loader = test_loader, 
    #                                         device = device,
    #                                         isbatch = True )
    # print(f"mean joints loss: {MeanJointsLoss}")

    # for joint, joint_groundtruth, mesh in zip(joints, joints_groundtruth, meshs):
    #     visualize_joints(joint.reshape(24,3), joint_groundtruth.reshape(24,3))
    #     # visualize_mesh(joint.reshape(24,3),  mesh)
    #     # visualize_mesh(joint_groundtruth.reshape(24,3),  mesh)
    #     break

    '''
    测试结果Nov.9.2023:
    mobilev2_smpl(best in 200epochs)
    mean joints Loss: 23.146158460889545
    forward代码执行时间: 236.95749974250793 秒
    平均一个iter执行时间: 4.999288490840367e-06
    mean joints loss: 23.146158460889545

    resnet18_smpl(best in 200epochs)
    mean joints Loss: 7.928131853376116
    forward代码执行时间: 197.83730554580688 秒
    平均一个iter执行时间: 3.6017979894365584e-06
    mean joints loss: 7.928131853376116
    '''
    # # 2. mesh 测试部分
    # MeanMeshLoss, meshs, meshs_groundtruth = eval_smpl_mesh(model=posenet,
    #                                         checkpoint_path = r'D:\workspace\python_ws\pose-master\checkpoints\best_resnet18_smpl.pth',
    #                                         test_loader = test_loader, 
    #                                         device = device,
    #                                         isbatch = True )

    # for mesh, mesh_groundtruth in zip(meshs, meshs_groundtruth):
    #     visualize_mesh(mesh, mesh_groundtruth)
    #     break

    # 3. 测你们全部
    # joints_pred, mesh_pred, joints_gt, mesh_gt, MPJPE, V2V = eval_smpl(model=pretrained_resnet18,
    #                                     checkpoint_path = '/root/pose_master/my_checkpoints/best_pretrained_resnet18_smpl.pth',
    #                                     test_loader = test_loader, 
    #                                     device = device,
    #                                     isbatch = False )
    
    # joints_pred, mesh_pred, joints_gt, mesh_gt, MPJPE, V2V = eval_smpl(model=pretrained_resnet50,
    #                                     checkpoint_path = '/root/pose_master/my_checkpoints/best_pretrained_resnet50_smpl.pth',
    #                                     test_loader = test_loader, 
    #                                     device = device,
    #                                     isbatch = False )
    
    # joints_pred, mesh_pred, joints_gt, mesh_gt, MPJPE, V2V = eval_smpl(model=pretrained_resnet18,
    #                                     checkpoint_path = '/root/pose_master/checkpoint_folder/loss_choose/best_pretrained_resnet18_smpl(256)0.04439585283398628.pth',
    #                                     test_loader = test_loader, 
    #                                     device = device,
    #                                     isbatch = False )
    # joints_pred, mesh_pred, joints_gt, mesh_gt, MPJPE, V2V = eval_smpl(model=pretrained_resnet18,
    #                                     checkpoint_path = '/root/pose_master/checkpoint_folder/loss_choose/best_pretrained_resnet18_smpl(256)l2.pth',
    #                                     test_loader = test_loader, 
    #                                     device = device,
    #                                     isbatch = False )
    # joints_pred, mesh_pred, joints_gt, mesh_gt, MPJPE, V2V = eval_smpl(model=mobile_with_mlp,
    #                                     checkpoint_path = '/root/pose_master/my_checkpoints/exp2/best_mobienetv2_with_mlp_refine.pth',
    #                                     test_loader = test_loader, 
    #                                     device = device,
    #                                     isbatch = False )