import torch
import torch.nn as nn
import sys
import pickle
import numpy as np
import cv2

# 将模块所在的目录添加到模块搜索路径
module_location = 'D:\workspace\python_ws\pose-master'  # 将此路径替换为实际的模块所在目录
sys.path.append(module_location)
from smpl.smpl_torch import SMPLModel

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
    def __init__(self, block, layers, num_classes=10+72):
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
        self.name = 'resnet+smpl'

        # 初始化对应性别的SMPL模型
        
        
        model_path_f = r'smpl\basicModel_f_lbs_10_207_0_v1.0.0.pkl'
        with open(model_path_f, 'rb') as f:
            human_f = pickle.load(f, encoding='latin1')
        self.J_regressor_f = torch.from_numpy(np.array(human_f['J_regressor'].todense())).type(torch.float32)
        self.weights_f = torch.from_numpy(human_f['weights']).type(torch.float32)
        self.posedirs_f = torch.from_numpy(human_f['posedirs']).type(torch.float32)
        self.v_template_f = torch.from_numpy(human_f['v_template']).type(torch.float32)
        self.shapedirs_f = torch.from_numpy(human_f['shapedirs'].r).type(torch.float32)
        self.kintree_table_f = human_f['kintree_table']
        self.faces_f = human_f['f']

        model_path_m = r'smpl\basicmodel_m_lbs_10_207_0_v1.0.0.pkl'
        with open(model_path_m, 'rb') as f:
            human_m = pickle.load(f, encoding='latin1')
        self.J_regressor_m = torch.from_numpy(np.array(human_m['J_regressor'].todense())).type(torch.float32)
        self.weights_m = torch.from_numpy(human_m['weights']).type(torch.float32)
        self.posedirs_m = torch.from_numpy(human_m['posedirs']).type(torch.float32)
        self.v_template_m = torch.from_numpy(human_m['v_template']).type(torch.float32)
        self.shapedirs_m = torch.from_numpy(human_m['shapedirs'].r).type(torch.float32)
        self.kintree_table_m = human_m['kintree_table']
        self.faces_m = human_f['f']

    def posemap(p):
        p = p.ravel()[3:]   # 跳过根结点
        return np.concatenate([(cv2.Rodrigues(np.array(pp))[0]-np.eye(3)).ravel() for pp in p.reshape((-1,3))]).ravel()

    def forward_smpl_f(self, betas, pose, trans, simplify=False):
        id_to_col = {self.kintree_table[1, i]: i
            for i in range(self.kintree_table.shape[1])}
        parent = {
            i: id_to_col[self.kintree_table[0, i]]
            for i in range(1, self.kintree_table.shape[1])
        }
        v_shaped = torch.tensordot(self.shapedirs, betas, dims=([2], [0])) + self.v_template
        J = torch.matmul(self.J_regressor, v_shaped)
        R_cube_big = self.rodrigues(pose.view(-1, 1, 3))

        if simplify:
            v_posed = v_shaped
        else:
            R_cube = R_cube_big[1:]
            I_cube = (torch.eye(3, dtype=torch.float32).unsqueeze(dim=0) + \
                torch.zeros((R_cube.shape[0], 3, 3), dtype=torch.float32)).to(self.device)
            lrotmin = torch.reshape(R_cube - I_cube, (-1, 1)).squeeze()
            v_posed = v_shaped + torch.tensordot(self.posedirs, lrotmin, dims=([2], [0]))

        results = []
        results.append(
        self.with_zeros(torch.cat((R_cube_big[0], torch.reshape(J[0, :], (3, 1))), dim=1))
        )

        for i in range(1, self.kintree_table.shape[1]):
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
        T = torch.tensordot(self.weights, results, dims=([1], [0]))
        rest_shape_h = torch.cat(
        (v_posed, torch.ones((v_posed.shape[0], 1), dtype=torch.float32).to(self.device)), dim=1
        )
        v = torch.matmul(T, torch.reshape(rest_shape_h, (-1, 4, 1)))
        v = torch.reshape(v, (-1, 4))[:, :3]
        result = v + torch.reshape(trans, (1, 3))
        # print(result.shape)
        # print(self.J_regressor.shape)
        joints = torch.tensordot(result, self.J_regressor, dims=([0], [1])).transpose(0, 1)
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
            if genders[i].int() == 0:
                mesh, joint = self.smpl_f(betas = batch[:10], pose = batch[10:],trans = trans[i])
            elif genders[i].int() == 1:
                mesh, joint = self.smpl_m(betas = batch[:10], pose = batch[10:],trans = trans[i])

            meshs.append(mesh)
            joints.append(torch.reshape(joint,(1,72)))

        meshs = torch.cat(meshs, dim = 0)
        joints = torch.cat(joints, dim = 0)
        # print(joints.double())
        # print(joints.shape)

        return joints



def posenet(num_classes=10+72):#默认直接预测出24×3的关节点位置

    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)



if __name__ == "__main__":
# 创建ResNet-18模型
    # 创建ResNet模型实例
    
    model = posenet(10+72) # 10个shape参数和 24*3的pose参数

    # smpl = SMPLModel()
    # 打印模型结构
    print(model)
