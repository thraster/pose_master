import numpy as np
import pickle
import torch
from torch.nn import Module
import os
from time import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import MultipleLocator

class SMPLModel(Module):
  def __init__(self, device=None, model_path='./model.pkl'):
    
    super(SMPLModel, self).__init__()
    with open(model_path, 'rb') as f:
      params = pickle.load(f, encoding='latin1')

    self.J_regressor = torch.from_numpy(
      np.array(params['J_regressor'].todense())
    ).type(torch.float32)
    if 'joint_regressor' in params.keys():
      self.joint_regressor = torch.from_numpy(
        np.array(params['joint_regressor'].T.todense())
      ).type(torch.float32)
    else:
      self.joint_regressor = torch.from_numpy(
        np.array(params['J_regressor'].todense())
      ).type(torch.float32)

    # J_regressor: (24, 6890), 与 vertices (6890, 3) 相乘边得到 joints 位置 (24, 3)
    # f: (13776, 3)，faces，我们说到 mesh 除了有 vertices 组成，还有一个 triangle list，每个 triangle 由三个 vertices index 组成
    # kintree_table: (2, 24)，一般取第一行，这就是上面提到的每个点的父节点
    # weights: (6890, 24), blend weights, 定义了顶点受每个 joint 旋转矩阵影响的权重
    # shapedirs: (6890, 3, 10), 表示体型参数到 shape blend shape 的映射关系
    # posedirs: (6890, 3, 207), 表示姿势参数到 pose blend shape 的映射关系
    # v_template: (6890, 3), 人体基模版的 vertices

    self.weights = torch.from_numpy(params['weights']).type(torch.float32)
    self.posedirs = torch.from_numpy(params['posedirs']).type(torch.float32)
    self.v_template = torch.from_numpy(params['v_template']).type(torch.float32)
    self.shapedirs = torch.from_numpy(params['shapedirs'].r).type(torch.float32)
    self.kintree_table = params['kintree_table']
    self.faces = params['f']
    self.device = device if device is not None else torch.device('cpu')
    for name in ['J_regressor', 'joint_regressor', 'weights', 'posedirs', 'v_template', 'shapedirs']:
      _tensor = getattr(self, name)
      # print(' Tensor {} shape: '.format(name), _tensor.shape)
      setattr(self, name, _tensor.to(device))

  @staticmethod
  def rodrigues(r):
    """
    Rodrigues' rotation formula that turns axis-angle tensor into rotation
    matrix in a batch-ed manner.

    Parameter:
    ----------
    r: Axis-angle rotation tensor of shape [batch_size * angle_num, 1, 3].

    Return:
    -------
    Rotation matrix of shape [batch_size * angle_num, 3, 3].

    """
    eps = r.clone().normal_(std=1e-8)
    theta = torch.norm(r + eps, dim=(1, 2), keepdim=True)  # dim cannot be tuple
    theta_dim = theta.shape[0]
    r_hat = r / theta
    cos = torch.cos(theta)
    z_stick = torch.zeros(theta_dim, dtype=torch.float32).to(r.device)
    m = torch.stack(
      (z_stick, -r_hat[:, 0, 2], r_hat[:, 0, 1], r_hat[:, 0, 2], z_stick,
       -r_hat[:, 0, 0], -r_hat[:, 0, 1], r_hat[:, 0, 0], z_stick), dim=1)
    m = torch.reshape(m, (-1, 3, 3))
    i_cube = (torch.eye(3, dtype=torch.float32).unsqueeze(dim=0) \
             + torch.zeros((theta_dim, 3, 3), dtype=torch.float32)).to(r.device)
    A = r_hat.permute(0, 2, 1)
    dot = torch.matmul(A, r_hat)
    R = cos * i_cube + (1 - cos) * dot + torch.sin(theta) * m
    return R

  @staticmethod
  def with_zeros(x):
    """
    Append a [0, 0, 0, 1] tensor to a [3, 4] tensor.

    Parameter:
    ---------
    x: Tensor to be appended.

    Return:
    ------
    Tensor after appending of shape [4,4]

    """
    ones = torch.tensor(
      [[[0.0, 0.0, 0.0, 1.0]]], dtype=torch.float32
    ).expand(x.shape[0],-1,-1).to(x.device)
    ret = torch.cat((x, ones), dim=1)
    return ret

  @staticmethod
  def pack(x):
    """
    Append zero tensors of shape [4, 3] to a batch of [4, 1] shape tensor.

    Parameter:
    ----------
    x: A tensor of shape [batch_size, 4, 1]

    Return:
    ------
    A tensor of shape [batch_size, 4, 4] after appending.

    """
    zeros43 = torch.zeros(
      (x.shape[0], x.shape[1], 4, 3), dtype=torch.float32).to(x.device)
    ret = torch.cat((zeros43, x), dim=3)
    return ret

  def write_obj(self, verts, file_name):
    with open(file_name, 'w') as fp:
      for v in verts:
        fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))

      for f in self.faces + 1:
        fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))

  def forward(self, betas, pose, trans, simplify=False):
    
    """
          Construct a compute graph that takes in parameters and outputs a tensor as
          model vertices. Face indices are also returned as a numpy ndarray.
          
          20190128: Add batch support.

          Parameters:
          ---------
          pose: Also known as 'theta', an [N, 24, 3] tensor indicating child joint rotation
          relative to parent joint. For root joint it's global orientation.
          Represented in a axis-angle format.

          betas: Parameter for model shape. A tensor of shape [N, 10] as coefficients of
          PCA components. Only 10 components were released by SMPL author.

          trans: Global translation tensor of shape [N, 3].

          Return:
          ------
          A 3-D tensor of [N * 6890 * 3] for vertices,
          and the corresponding [N * 19 * 3] joint positions.

    """
    batch_num = betas.shape[0]
    id_to_col = {self.kintree_table[1, i]: i
                 for i in range(self.kintree_table.shape[1])}
    parent = {
      i: id_to_col[self.kintree_table[0, i]]
      for i in range(1, self.kintree_table.shape[1])
    }
    v_shaped = torch.tensordot(betas, self.shapedirs, dims=([1], [2])) + self.v_template
    J = torch.matmul(self.J_regressor, v_shaped)
    R_cube_big = self.rodrigues(pose.view(-1, 1, 3)).reshape(batch_num, -1, 3, 3)

    if simplify:
      v_posed = v_shaped
    else:
      R_cube = R_cube_big[:, 1:, :, :]
      I_cube = (torch.eye(3, dtype=torch.float32).unsqueeze(dim=0) + \
        torch.zeros((batch_num, R_cube.shape[1], 3, 3), dtype=torch.float32)).to(self.device)
      lrotmin = (R_cube - I_cube).reshape(batch_num, -1, 1).squeeze(dim=2)
      v_posed = v_shaped + torch.tensordot(lrotmin, self.posedirs, dims=([1], [2]))

    results = []
    results.append(
      self.with_zeros(torch.cat((R_cube_big[:, 0], torch.reshape(J[:, 0, :], (-1, 3, 1))), dim=2))
    )
    for i in range(1, self.kintree_table.shape[1]):
      results.append(
        torch.matmul(
          results[parent[i]],
          self.with_zeros(
            torch.cat(
              (R_cube_big[:, i], torch.reshape(J[:, i, :] - J[:, parent[i], :], (-1, 3, 1))),
              dim=2
            )
          )
        )
      )
    
    stacked = torch.stack(results, dim=1)
    results = stacked - \
      self.pack(
        torch.matmul(
          stacked,
          torch.reshape(
            torch.cat((J, torch.zeros((batch_num, 24, 1), dtype=torch.float32).to(self.device)), dim=2),
            (batch_num, 24, 4, 1)
          )
        )
      )
    # Restart from here
    T = torch.tensordot(results, self.weights, dims=([1], [1])).permute(0, 3, 1, 2)
    rest_shape_h = torch.cat(
      (v_posed, torch.ones((batch_num, v_posed.shape[1], 1), dtype=torch.float32).to(self.device)), dim=2
    )
    v = torch.matmul(T, torch.reshape(rest_shape_h, (batch_num, -1, 4, 1)))
    v = torch.reshape(v, (batch_num, -1, 4))[:, :, :3]
    result = v + torch.reshape(trans, (batch_num, 1, 3))
    # estimate 3D joint locations
    # print(result.shape)
    # print(self.joint_regressor.shape)
    joints = torch.tensordot(result, self.joint_regressor, dims=([1], [1])).transpose(1, 2)
    return result, joints


def test_smpl(device, pose, betas, trans):
  smpl = SMPLModel(device=device, model_path= r'D:\workspace\python_ws\pose-master\smpl\basicModel_f_lbs_10_207_0_v1.0.0.pkl')

  s = time()

  result, joints = smpl(betas, pose, trans)
  # print("mesh:",result.shape)
  # print("joint:", joints.shape)
  print("time cost:",time() - s)
  return result, joints
  # outmesh_path = './smpl_torch_{}.obj'
  # for i in range(result.shape[0]):
      # model.write_obj(result[i], outmesh_path.format(i))

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
    labels = ['markers_xyz', 'smpl_joints']
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


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import sys
    module_location = r'D:\workspace\python_ws\pose-master'
    sys.path.append(module_location)
    from load_dataset_lmdb import SkeletonDatasetLMDB
    lmdb_path = r'D:\workspace\python_ws\pose-master\dataset\lmdb_data_test'  # 替换为LMDB数据库的路径
    lmdb_dataset = SkeletonDatasetLMDB(lmdb_path,transform=True)
    # lmdb_dataset = SkeletonDatasetLMDB(lmdb_path,transform=None)
    # 获取数据集的长度
    print("Dataset length:", len(lmdb_dataset))

    # 加载数据
    test_loader = torch.utils.data.DataLoader(lmdb_dataset, batch_size=4, shuffle=False,num_workers = 0)

    if torch.cuda.is_available():
      device = torch.device('cuda')
    else:
      device = torch.device('cpu')

    # pose_size = 72
    # beta_size = 10

    # np.random.seed(9608)
    # batch_size = 64
    # pose = torch.from_numpy((np.random.rand(batch_size, pose_size) - 0.5) * 0.4)\
    #       .type(torch.float32).to(device)
    # betas = torch.from_numpy((np.random.rand(batch_size, beta_size) - 0.5) * 0.06) \
    #         .type(torch.float32).to(device)
    # trans = torch.from_numpy(np.ones((batch_size, 3))).type(torch.float32).to(device)

    # print(pose.shape,betas.shape,trans.shape)
    for data_item in test_loader:
      pose = data_item['pose'].to(device)
      betas = data_item['shape'].to(device)
      trans = (data_item['trans']-torch.tensor([0.0,0.0,0.075])).to(device)
      skeleton = data_item['skeleton']
      break
    mesh, joint = test_smpl(device,pose,betas,trans)
    print(mesh.shape,joint.shape)


    for batch_idx in range(0, mesh.size(0)):
      # # 假设您有一个批次的mesh和joint
      # batch_idx = 0  # 选择要显示的批次索引

      # 可视化mesh
      mesh_data = mesh[batch_idx].cpu().numpy()

      # 可视化joint
      joint_data = joint[batch_idx].cpu().numpy()
      # visualize_joints(skeleton[batch_idx].reshape(24,3),joint_data)
      # visualize_mesh(skeleton[batch_idx].reshape(24,3)-np.array([0.0, -0.25, 0.0375]),joint_data)

      # visualize_mesh(skeleton[batch_idx].reshape(24,3),joint_data-np.array([ -0.01413382 ,-0.21930961,  0.0170392]))


      division = joint_data-(skeleton[batch_idx].reshape(24,3)-torch.tensor([0.0,0.0,0.075])).numpy()
      # # trans_shift = trans[batch_idx].cpu().numpy()
      print(division)
      # # print(trans_shift-np.array([0.6,1.2,0.1]))
      print(f"division mean: {np.mean(division, axis=0)}")