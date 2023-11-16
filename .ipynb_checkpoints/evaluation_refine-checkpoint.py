import torch
import torch.nn as nn
import torch.optim as optim
import time
import sys
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import MultipleLocator

from models.mobilenet_refine import mobilenet_refine

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


def eval_smpl(model, checkpoint_path, test_loader, device, isbatch=True, gender = 'm'):
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

            skeletons = data['skeleton'].to(device)
            images = data['image'].to(device)
            genders = data['gender'].to(device)
            transs = data['trans'].to(device)
            shapes = data['shape'].to(device)
            poses = data['pose'].to(device)
            # print(trans.shape)
            # break
           
            start_time = time.time()

            feature_maps = model(images)

            forward_time = time.time()

            pred_shapes = F.adaptive_avg_pool2d(feature_maps[-3], (1, 1)).squeeze()
            pred_poses = F.adaptive_avg_pool2d(feature_maps[-2], (1, 1)).squeeze()
            pred_transs = F.adaptive_avg_pool2d(feature_maps[-1], (1, 1)).squeeze()

            if gender == 'm':
                mesh, joints = smpl_m(
                pose = pred_poses ,
                betas = pred_shapes,
                trans = pred_transs
                )

                mesh_gt, joints_gt = smpl_m(
                pose = poses ,
                betas = shapes,
                trans = transs 
                )

            elif gender == 'f':
                mesh, joints = smpl_f(
                pose = pred_poses ,
                betas = pred_shapes,
                trans = pred_transs
                )
                mesh_gt, joints_gt = smpl_f(
                pose = poses ,
                betas = shapes,
                trans = transs 
                )


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
        
        print(batch_size)
        print(i+1)
        # 输出测试结果
        MPJPE = totalMPJPE / batch_size/ batch_num
        V2V = totalV2V / batch_size/ batch_num
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
    test_data = SkeletonDatasetLMDB('/root/pose_master/dataset/test_lmdb_gt_m',  transform = True)
    batch_size = 256
    # shuffle要为true才能正常跑
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True,num_workers = 0, pin_memory=True)

    joints_pred, mesh_pred, joints_gt, mesh_gt, MPJPE, V2V = eval_smpl(model=mobilenet_refine,
                                        checkpoint_path = '/root/pose_master/my_checkpoints/refine/best_Mobilnet_refine_mod2(joint_supervized)_m.pth',
                                        test_loader = test_loader, 
                                        device = device,
                                        isbatch = False,
                                        gender = 'm' )

    test_data = SkeletonDatasetLMDB('/root/pose_master/dataset/test_lmdb_gt_f',  transform = True)

    # shuffle要为true才能正常跑
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True,num_workers = 0, pin_memory=True)

    joints_pred, mesh_pred, joints_gt, mesh_gt, MPJPE, V2V = eval_smpl(model=mobilenet_refine,
                                        checkpoint_path = '/root/pose_master/my_checkpoints/refine/best_Mobilnet_refine_mod2(joint_supervized)_f.pth',
                                        test_loader = test_loader, 
                                        device = device,
                                        isbatch = False,
                                        gender = 'f' )
    # for joint, joint_groundtruth in zip(joints_pred, joints_gt):
    #     visualize_joints(joint.reshape(24,3), joint_groundtruth.reshape(24,3))
    #     # visualize_mesh(joint.reshape(24,3),  mesh)
    #     # visualize_mesh(joint_groundtruth.reshape(24,3),  mesh)
    #     break