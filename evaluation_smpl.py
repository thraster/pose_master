import torch
import torch.nn as nn
import torch.optim as optim
import time
import sys

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import MultipleLocator

from models.resnet_smpl_batch import posenet
from models.mobilenet_smpl_batch import mobilenet
from smpl.smpl_torch_batch import SMPLModel



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
    model_path_f = r'D:\workspace\python_ws\pose-master\smpl\basicModel_f_lbs_10_207_0_v1.0.0.pkl'
    smpl_f = SMPLModel(device=device,model_path=model_path_f).to(device)

    model_path_m = r'D:\workspace\python_ws\pose-master\smpl\basicmodel_m_lbs_10_207_0_v1.0.0.pkl'
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
            transs = (data['trans']-torch.tensor([0.286,0.136,0])*4).to(device)
            transs = torch.stack([
                                data['trans'][:, 0] - 0.286,
                                data['trans'][:, 1] - 0.286 + 0.15,
                                0.12 - data['trans'][:, 2]
                            ], dim=1).to(device)
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

            processed_mesh = torch.empty((batch_size, 6890, 3)).to(device)  # 创建与x相同大小的空张量
            processed_mesh[genders.squeeze() == 0] = mesh_f  # 将x_f数据放回原始位置
            processed_mesh[genders.squeeze() == 1] = mesh_m  # 将x_m数据放回原始位置

            processed_joints = torch.empty((batch_size, 24, 3)).to(device)  # 创建与x相同大小的空张量
            processed_joints[genders.squeeze() == 0] = joints_f  # 将x_f数据放回原始位置
            processed_joints[genders.squeeze() == 1] = joints_m  # 将x_m数据放回原始位置


            start_time = time.time()
            # 前向传播
            mesh, joints = model(images, genders)
            forward_time = time.time()

            # 计算损失
            
            batchMeanMeshLoss = criterion(mesh, processed_mesh)

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

    return MeanMeshLoss, processed_mesh.cpu(), mesh.cpu()   # gt, pred

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
        totalMeanJointsLoss = 0.0
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

            batchMeanJointsLoss = criterion(joints*1000, skeletons*1000)

            totalMeanJointsLoss += batchMeanJointsLoss.item()

            total_time += forward_time-start_time

            if isbatch == True:
                
                print(f"forward代码执行时间: {total_time} 秒")
                print(f"平均一个iter执行时间: {(forward_time-start_time)/batch_size}")
                break
            else:
                batch_lenth = len(test_loader)
                print(f"evaluating... [{i}/{batch_num}]", end='\r')
                
        # 输出测试结果
        MeanJointsLoss = totalMeanJointsLoss / batch_size/ (i+1)
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


# 用法示例
if __name__ == "__main__":

    
    from load_dataset_lmdb import SkeletonDatasetLMDB

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    test_data = SkeletonDatasetLMDB(r'D:\workspace\python_ws\pose-master\dataset\test_lmdb_new',  transform = True)

    batch_size = 16 # 

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True,num_workers = 0, pin_memory=True)

    # 1. joint 测试部分
    MeanJointsLoss, joints, joints_groundtruth,meshs = eval_smpl_joints(model=posenet,
                                            checkpoint_path = r'D:\workspace\python_ws\pose-master\checkpoints\best_resnet18_smpl.pth',
                                            test_loader = test_loader, 
                                            device = device,
                                            isbatch = False )
    print(f"mean joints loss: {MeanJointsLoss}")

    for joint, joint_groundtruth, mesh in zip(joints, joints_groundtruth, meshs):
        visualize_joints(joint.reshape(24,3), joint_groundtruth.reshape(24,3))
        # visualize_mesh(joint.reshape(24,3),  mesh)
        # visualize_mesh(joint_groundtruth.reshape(24,3),  mesh)
        break
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
    # # # 2. mesh 测试部分
    # MeanJointsLoss, meshs, meshs_groundtruth = eval_smpl_mesh(model=mobilenet,
    #                                         checkpoint_path = r'D:\workspace\python_ws\pose-master\checkpoints\best_mobilenetv2_smpl.pth',
    #                                         test_loader = test_loader, 
    #                                         device = device,
    #                                         isbatch = True )

    # for mesh, mesh_groundtruth in zip(meshs, meshs_groundtruth):
    #     visualize_mesh(mesh, mesh_groundtruth)
    #     break
