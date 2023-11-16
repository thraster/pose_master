import torch
import torch.nn as nn
# import load_dataset
import time
import sys
import torch.nn.functional as F
# 将模块所在的目录添加到模块搜索路径
module_location = '/root/pose_master'  # 将此路径替换为实际的模块所在目录
sys.path.append(module_location)
from smpl.smpl_torch_batch import SMPLModel

min_shape = -3
max_shape = 3
min_pose = -2.8452
max_pose = 4.1845
min_trans = -0.0241
max_trans = 1.5980





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


def test_model_refine(model, test_loader, device, gender):
    '''
    ***使用load_dataset_lmdb 不能使用使用load_dataset_lmdb_mod1***
    
    test函数,用于一个epoch结束后的test,并保存该epoch的checkpoint

    model:被测试的模型
    epoch:epoch数
    test_laoder:测试集的dataloader实例
    device:运行的设备
    criterion:损失函数
    
    '''
    
    # 将模型设置为评估模式
    criterion = nn.L1Loss()
    model_path_f = '/root/pose_master/smpl/basicModel_f_lbs_10_207_0_v1.0.0.pkl'
    smpl_f = SMPLModel(device=device,model_path=model_path_f).to(device)

    model_path_m ='/root/pose_master/smpl/basicmodel_m_lbs_10_207_0_v1.0.0.pkl'
    smpl_m = SMPLModel(device=device,model_path=model_path_m).to(device)
    
    model.eval()
    print(f'testing... [{model.name}_{gender}]')
    # 定义损失函数（根据您的任务选择合适的损失函数）
    # criterion = nn.MSELoss()  # 示例中使用了二分类交叉熵损失
    batch_num = len(test_loader)

    start_time = time.time()
    with torch.no_grad():
        shape_loss = [0,0]
        pose_loss = [0,0]
        trans_loss = [0,0]
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



            feature_maps = model(images)
            
            
            # 1. loss of pred_params
            shape_loss_batch = []
            pose_loss_batch = []
            trans_loss_batch = []


            for loss_idx in range(0,len(feature_maps) // 3):
                pred_shapes = F.adaptive_avg_pool2d(feature_maps[loss_idx*3], (1, 1)).squeeze()
                pred_poses = F.adaptive_avg_pool2d(feature_maps[loss_idx*3 + 1], (1, 1)).squeeze()
                pred_transs = F.adaptive_avg_pool2d(feature_maps[loss_idx*3 + 2], (1, 1)).squeeze()


                shape_loss_batch.append(criterion(pred_shapes, shapes))
                pose_loss_batch.append(criterion(pred_poses, poses))
                trans_loss_batch.append(criterion(pred_transs, transs))

            
            loss = 0.0
            for loss_idx in range(0,len(feature_maps) // 3):
                shape_loss[loss_idx] += shape_loss_batch[loss_idx].item()
                pose_loss[loss_idx] += pose_loss_batch[loss_idx].item()
                trans_loss[loss_idx] += trans_loss_batch[loss_idx].item()




            pred_shapes = F.adaptive_avg_pool2d(feature_maps[-3], (1, 1)).squeeze()
            pred_poses = F.adaptive_avg_pool2d(feature_maps[-2], (1, 1)).squeeze()
            pred_transs = F.adaptive_avg_pool2d(feature_maps[-1], (1, 1)).squeeze()

            # 2. loss of pred_joints, pred_mesh
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
            batchMPJPE = calculate_mpjpe((joints_gt*1000), (joints*1000))

            batchV2V = calculate_v2v_error((mesh_gt*1000), (mesh*1000))


            totalMPJPE += torch.sum(batchMPJPE)
            totalV2V += torch.sum(batchV2V)


            print(f"testing... [{i}/{batch_num}]", end='\r')

    # 输出测试结果
    for i,loss in enumerate(shape_loss):
        print(f"shape loss[{i}] = {loss / batch_num:.6f}")
    for i,loss in enumerate(pose_loss):
        print(f"pose loss[{i}] = {loss / batch_num:.6f}")
    for i,loss in enumerate(trans_loss):
        print(f"trans loss[{i}] = {loss / batch_num:.6f}")
    MPJPE = totalMPJPE / batch_size/ batch_num
    V2V = totalV2V / batch_size/ batch_num
    print(f"Mean Per-Joint Position Error (MPJPE): {MPJPE:.6f}")
    print(f"vertex-to-vertex error (v2v): {V2V:.6f}")
    forward_time = time.time()
    total_time = forward_time-start_time
    print(f"testing time: {total_time:.6f}s")
    # print(f"mean mesh loss = {MeanMeshLoss / lenth}")

    return MPJPE, V2V, shape_loss[1],shape_loss[0],pose_loss[1],pose_loss[0],trans_loss[1],trans_loss[0]

def test_model_refine_mod1(model, test_loader, device, gender):
    '''
    ***使用load_dataset_lmdb_mod1***
    
    test函数,用于一个epoch结束后的test,并保存该epoch的checkpoint

    model:被测试的模型
    epoch:epoch数
    test_laoder:测试集的dataloader实例
    device:运行的设备
    criterion:损失函数
    
    '''
    
    # 将模型设置为评估模式
    criterion = nn.L1Loss()
    model_path_f = '/root/pose_master/smpl/basicModel_f_lbs_10_207_0_v1.0.0.pkl'
    smpl_f = SMPLModel(device=device,model_path=model_path_f).to(device)

    model_path_m ='/root/pose_master/smpl/basicmodel_m_lbs_10_207_0_v1.0.0.pkl'
    smpl_m = SMPLModel(device=device,model_path=model_path_m).to(device)
    
    model.eval()
    print(f'testing... [{model.name}_{gender}]')
    # 定义损失函数（根据您的任务选择合适的损失函数）
    # criterion = nn.MSELoss()  # 示例中使用了二分类交叉熵损失
    batch_num = len(test_loader)

    start_time = time.time()
    with torch.no_grad():
        shape_loss = [0,0]
        pose_loss = [0,0]
        trans_loss = [0,0]
        totalMPJPE = 0.0
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



            feature_maps = model(images)
            
            
            # 1. loss of pred_params
            shape_loss_batch = []
            pose_loss_batch = []
            trans_loss_batch = []


            for loss_idx in range(0,len(feature_maps) // 3):
                pred_shapes = F.adaptive_avg_pool2d(feature_maps[loss_idx*3], (1, 1)).squeeze()
                pred_poses = F.adaptive_avg_pool2d(feature_maps[loss_idx*3 + 1], (1, 1)).squeeze()
                pred_transs = F.adaptive_avg_pool2d(feature_maps[loss_idx*3 + 2], (1, 1)).squeeze()

                pred_shapes = (pred_shapes - min_shape) / (max_shape - min_shape)
                pred_poses = (pred_poses - min_pose) / (max_pose - min_pose)
                pred_transs = (pred_transs - min_trans) / (max_trans - min_trans)

                # print(pred_shapes.shape, pred_poses.shape, pred_transs.shape)
                shape_loss_batch.append(criterion(pred_shapes, shapes))
                pose_loss_batch.append(criterion(pred_poses, poses))
                trans_loss_batch.append(criterion(pred_transs, transs))

                # losses.append(criterion(pred_shapes, shapes))
                # losses.append(criterion(pred_poses, poses))
                # losses.append(criterion(pred_transs, transs))
            
            for loss_idx in range(0,len(feature_maps) // 3):
                shape_loss[loss_idx] += shape_loss_batch[loss_idx].item()
                pose_loss[loss_idx] += pose_loss_batch[loss_idx].item()
                trans_loss[loss_idx] += trans_loss_batch[loss_idx].item()

            pred_shapes = F.adaptive_avg_pool2d(feature_maps[-3], (1, 1)).squeeze()
            pred_poses = F.adaptive_avg_pool2d(feature_maps[-2], (1, 1)).squeeze()
            pred_transs = F.adaptive_avg_pool2d(feature_maps[-1], (1, 1)).squeeze()

            # 2. loss of pred_joints
            if gender == 'm':
                mesh, joints = smpl_m(
                pose = pred_poses ,
                betas = pred_shapes,
                trans = pred_transs
                )

            elif gender == 'f':
                mesh, joints = smpl_f(
                pose = pred_poses ,
                betas = pred_shapes,
                trans = pred_transs
                )

            # 计算损失
            batchMPJPE = calculate_mpjpe((skeletons*1000), (joints*1000))

            totalMPJPE += torch.sum(batchMPJPE)


            print(f"testing... [{i}/{batch_num}]", end='\r')

    # 输出测试结果
    for i,loss in enumerate(shape_loss):
        print(f"shape loss[{i}] = {loss / batch_num:.6f}")
    for i,loss in enumerate(pose_loss):
        print(f"pose loss[{i}] = {loss / batch_num:.6f}")
    for i,loss in enumerate(trans_loss):
        print(f"trans loss[{i}] = {loss / batch_num:.6f}")
    MPJPE = totalMPJPE / batch_size/ batch_num
    print(f"Mean Per-Joint Position Error (MPJPE): {MPJPE:.6f}")
    forward_time = time.time()
    total_time = forward_time-start_time
    print(f"testing time: {total_time:.6f}s")
    # print(f"mean mesh loss = {MeanMeshLoss / lenth}")

    return MPJPE, shape_loss[1],shape_loss[0],pose_loss[1],pose_loss[0],trans_loss[1],trans_loss[0]


if __name__ == "__main__":
    from models.mobilenet_refine import mobilenet_refine
    from load_dataset_lmdb_mod1 import SkeletonDatasetLMDB
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gender = 'm'
    batch_size = 256
    
    
    model = mobilenet_refine(device = device).to(device)
    checkpoint = torch.load(f'/root/pose_master/my_checkpoints/refine/last_Mobilnet_refine_mod2(joint_supervized)_{gender}.pth')
    # print(f"loading checkpoint [{checkpoint_path}] successed!")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_data = SkeletonDatasetLMDB(f'/root/pose_master/dataset/test_lmdb_gt_{gender}',True)
    
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True,num_workers = 0, pin_memory=True)
    
    test_model_refine(model = model, test_loader = test_loader, device = device, gender = gender)
