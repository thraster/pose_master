import torch
import torch.nn as nn
import sys
import pickle
import numpy as np
import cv2
import time

# 将模块所在的目录添加到模块搜索路径
module_location = 'D:\workspace\python_ws\pose-master'  # 将此路径替换为实际的模块所在目录
sys.path.append(module_location)
from smpl.smpl_torch import SMPLModel

# Torch.manual_seed(3407) is all you need
torch.manual_seed(3407)

# 定义ResNet基本块
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 下采样层
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        residual = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out

# 定义ResNet主网络
class ResNet_smpl(nn.Module):
    def __init__(self, block, layers, num_classes=10+72, device='cuda'):
        super(ResNet_smpl, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.make_layer(block, 64, layers[0])
        self.layer2 = self.make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self.make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self.make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.name = 'resnet+smpl(embeded)'
        self.device = device
        # 初始化对应性别的SMPL模型
        
        
        model_path_f = r'smpl\basicModel_f_lbs_10_207_0_v1.0.0.pkl'
        with open(model_path_f, 'rb') as f:
            human_f = pickle.load(f, encoding='latin1')
        self.J_regressor_f = torch.from_numpy(np.array(human_f['J_regressor'].todense())).type(torch.float32).to(device)
        self.weights_f = torch.from_numpy(human_f['weights']).type(torch.float32).to(device)
        self.posedirs_f = torch.from_numpy(human_f['posedirs']).type(torch.float32).to(device)
        self.v_template_f = torch.from_numpy(human_f['v_template']).type(torch.float32).to(device)
        self.shapedirs_f = torch.from_numpy(human_f['shapedirs'].r).type(torch.float32).to(device)
        self.kintree_table_f = human_f['kintree_table']
        self.faces_f = human_f['f']

        model_path_m = r'smpl\basicmodel_m_lbs_10_207_0_v1.0.0.pkl'
        with open(model_path_m, 'rb') as f:
            human_m = pickle.load(f, encoding='latin1')
        self.J_regressor_m = torch.from_numpy(np.array(human_m['J_regressor'].todense())).type(torch.float32).to(device)
        self.weights_m = torch.from_numpy(human_m['weights']).type(torch.float32).to(device)
        self.posedirs_m = torch.from_numpy(human_m['posedirs']).type(torch.float32).to(device)
        self.v_template_m = torch.from_numpy(human_m['v_template']).type(torch.float32).to(device)
        self.shapedirs_m = torch.from_numpy(human_m['shapedirs'].r).type(torch.float32).to(device)
        self.kintree_table_m = human_m['kintree_table']
        self.faces_m = human_f['f']

    def posemap(p):
        p = p.ravel()[3:]   # 跳过根结点
        return np.concatenate([(cv2.Rodrigues(np.array(pp))[0]-np.eye(3)).ravel() for pp in p.reshape((-1,3))]).ravel()

    @staticmethod
    def rodrigues(r):
        """
        Rodrigues' rotation formula that turns axis-angle tensor into rotation
        matrix in a batch-ed manner.

        Parameter:
        ----------
        r: Axis-angle rotation tensor of shape [batch_size, 1, 3].

        Return:
        -------
        Rotation matrix of shape [batch_size, 3, 3].

        """
        #r = r.to(self.device)
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
        zeros43 = torch.zeros((x.shape[0], 4, 3), dtype=torch.float32).to(x.device)
        ret = torch.cat((zeros43, x), dim=2)
        return ret
    
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
        ones = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32).to(x.device)
        ret = torch.cat((x, ones), dim=0)
        return ret

    def forward_smpl_f(self, betas, pose, trans, simplify=False):
        id_to_col = {self.kintree_table_f[1, i]: i
            for i in range(self.kintree_table_f.shape[1])}
        parent = {
            i: id_to_col[self.kintree_table_f[0, i]]
            for i in range(1, self.kintree_table_f.shape[1])
        }
        v_shaped = torch.tensordot(self.shapedirs_f, betas, dims=([2], [0])) + self.v_template_f
        J = torch.matmul(self.J_regressor_f, v_shaped)
        R_cube_big = self.rodrigues(pose.view(-1, 1, 3))

        if simplify:
            v_posed = v_shaped
        else:
            R_cube = R_cube_big[1:]
            I_cube = (torch.eye(3, dtype=torch.float32).unsqueeze(dim=0) + \
                torch.zeros((R_cube.shape[0], 3, 3), dtype=torch.float32)).to(self.device)
            lrotmin = torch.reshape(R_cube - I_cube, (-1, 1)).squeeze()
            v_posed = v_shaped + torch.tensordot(self.posedirs_f, lrotmin, dims=([2], [0]))

        results = []
        results.append(
        self.with_zeros(torch.cat((R_cube_big[0], torch.reshape(J[0, :], (3, 1))), dim=1))
        )

        for i in range(1, self.kintree_table_f.shape[1]):
            results.append(
                torch.matmul(
                    results[parent[i]],
                    self.with_zeros(
                        torch.cat(
                        (R_cube_big[i], torch.reshape(J[i, :] - J[parent[i], :], (3, 1))),
                        dim=1
                        )
                    )
                )
            )

        stacked = torch.stack(results, dim=0)
        results = stacked - \
        self.pack(
            torch.matmul(
            stacked,
            torch.reshape(
                torch.cat((J, torch.zeros((24, 1), dtype=torch.float32).to(self.device)), dim=1),
                (24, 4, 1)
            )
            )
        )
        T = torch.tensordot(self.weights_f, results, dims=([1], [0]))
        rest_shape_h = torch.cat(
        (v_posed, torch.ones((v_posed.shape[0], 1), dtype=torch.float32).to(self.device)), dim=1
        )
        v = torch.matmul(T, torch.reshape(rest_shape_h, (-1, 4, 1)))
        v = torch.reshape(v, (-1, 4))[:, :3]
        result = v + torch.reshape(trans, (1, 3))
        # print(result.shape)
        # print(self.J_regressor.shape)
        joints = torch.tensordot(result, self.J_regressor_f, dims=([0], [1])).transpose(0, 1)
        return result, joints
    
    def forward_smpl_m(self, betas, pose, trans, simplify=False):
        id_to_col = {self.kintree_table_m[1, i]: i
            for i in range(self.kintree_table_m.shape[1])}
        parent = {
            i: id_to_col[self.kintree_table_m[0, i]]
            for i in range(1, self.kintree_table_m.shape[1])
        }
        v_shaped = torch.tensordot(self.shapedirs_m, betas, dims=([2], [0])) + self.v_template_m
        J = torch.matmul(self.J_regressor_m, v_shaped)
        R_cube_big = self.rodrigues(pose.view(-1, 1, 3))

        if simplify:
            v_posed = v_shaped
        else:
            R_cube = R_cube_big[1:]
            I_cube = (torch.eye(3, dtype=torch.float32).unsqueeze(dim=0) + \
                torch.zeros((R_cube.shape[0], 3, 3), dtype=torch.float32)).to(self.device)
            lrotmin = torch.reshape(R_cube - I_cube, (-1, 1)).squeeze()
            v_posed = v_shaped + torch.tensordot(self.posedirs_m, lrotmin, dims=([2], [0]))

        results = []
        results.append(
        self.with_zeros(torch.cat((R_cube_big[0], torch.reshape(J[0, :], (3, 1))), dim=1))
        )

        for i in range(1, self.kintree_table_m.shape[1]):
            results.append(
                torch.matmul(
                    results[parent[i]],
                    self.with_zeros(
                        torch.cat(
                        (R_cube_big[i], torch.reshape(J[i, :] - J[parent[i], :], (3, 1))),
                        dim=1
                        )
                    )
                )
            )

        stacked = torch.stack(results, dim=0)
        results = stacked - \
        self.pack(
            torch.matmul(
            stacked,
            torch.reshape(
                torch.cat((J, torch.zeros((24, 1), dtype=torch.float32).to(self.device)), dim=1),
                (24, 4, 1)
            )
            )
        )
        T = torch.tensordot(self.weights_m, results, dims=([1], [0]))
        rest_shape_h = torch.cat(
        (v_posed, torch.ones((v_posed.shape[0], 1), dtype=torch.float32).to(self.device)), dim=1
        )
        v = torch.matmul(T, torch.reshape(rest_shape_h, (-1, 4, 1)))
        v = torch.reshape(v, (-1, 4))[:, :3]
        result = v + torch.reshape(trans, (1, 3))
        # print(result.shape)
        # print(self.J_regressor.shape)
        joints = torch.tensordot(result, self.J_regressor_m, dims=([0], [1])).transpose(0, 1)
        return result, joints

    def make_layer(self, block, out_channels, blocks, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x, genders, trans):
        # print(genders[0])
        '''
        skeleton: torch.Size([batchsize, 72])
        image: torch.Size([batchsize, 1, 224, 224])
        gender: torch.Size([batchsize, 1])
        trans: torch.Size([batchsize, 3])
        
        '''
        meshs = []
        joints = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        for i, batch in enumerate(x):
            # print(batch.shape)
            # print(i)
            if genders[i,0].int() == 0:
                mesh, joint = self.forward_smpl_f(betas = batch[:10], pose = batch[10:],trans = trans[i])
            elif genders[i,0].int() == 1:
                mesh, joint = self.forward_smpl_m(betas = batch[:10], pose = batch[10:],trans = trans[i])
            meshs.append(mesh)
            joints.append(torch.reshape(joint,(1,72)))
        # print(f"smpl forward time: {smpl2-smpl1}")
        meshs = torch.cat(meshs, dim = 0)
        joints = torch.cat(joints, dim = 0)
        # print(joints.double())
        # print(joints.shape)

        return joints



def posenet(num_classes=10+72,device='cuda'):#默认直接预测出24×3的关节点位置

    return ResNet_smpl(BasicBlock, [2, 2, 2, 2], num_classes,device)



if __name__ == "__main__":
# 创建ResNet-18模型
    # 创建ResNet模型实例
    
    model = posenet(10+72) # 10个shape参数和 24*3的pose参数

    # smpl = SMPLModel()
    # 打印模型结构
    print(model)
