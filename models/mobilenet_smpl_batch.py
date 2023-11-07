import torch
import torch.nn as nn
import sys
import pickle
import numpy as np
import cv2
import time
from torch.nn import Module

module_location = r'D:\workspace\python_ws\pose-master'
sys.path.append(module_location)
from smpl.smpl_torch_batch import test_smpl
from smpl.smpl_torch_batch import SMPLModel

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        hidden_dim = in_channels * expand_ratio
        self.use_res_connect = stride == 1 and in_channels == out_channels

        layers = []
        if expand_ratio != 1:
            layers.append(nn.Conv2d(in_channels, hidden_dim, 1, 1, 0, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU6(inplace=True))
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_multiplier=1.0, input_size=224, inverted_residual_setting=None, device = 'cuda'):
        super(MobileNetV2, self).__init__()
        self.name = 'mobilenetv2_smpl'
        input_channel = 32
        last_channel = 1280
        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]
        # 输入通道数改为了1
        self.features = [nn.Conv2d(1, input_channel, 3, 2, 1, bias=False)]
        self.features.append(nn.BatchNorm2d(input_channel))
        self.features.append(nn.ReLU6(inplace=True))
        for t, c, n, s in inverted_residual_setting:
            output_channel = int(c * width_multiplier)
            for i in range(n):
                stride = s if i == 0 else 1
                self.features.append(InvertedResidual(input_channel, output_channel, stride, t))
                input_channel = output_channel
        self.features.append(nn.Conv2d(input_channel, last_channel, 1, 1, 0, bias=False))
        self.features.append(nn.BatchNorm2d(last_channel))
        self.features.append(nn.ReLU6(inplace=True))
        self.features = nn.Sequential(*self.features)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(last_channel, num_classes)
        )

        self.device = device
        # print(self.device)
        # 初始化对应性别的SMPL模型
        
        model_path_f = r'D:\workspace\python_ws\pose-master\smpl\basicModel_f_lbs_10_207_0_v1.0.0.pkl'
        self.smpl_f = SMPLModel(device=self.device,model_path=model_path_f).to(device)

        model_path_m = r'D:\workspace\python_ws\pose-master\smpl\basicmodel_m_lbs_10_207_0_v1.0.0.pkl'
        self.smpl_m = SMPLModel(device=self.device,model_path=model_path_m).to(device)

    def forward(self, x, genders=None):
        '''
        全连接层输出长度为85的X
        x[0:10]: betas
        x[10:82]: pose 
        x[82:85]: trans
        ----------
        x[10:12]描述了根节点的rotation
        x[82:85]描述了更节点的global translation
        '''
        '''
        1. mobilenet forward
        '''
        x = self.features(x)
        x = self.classifier(x)
        '''
        2. smpl forward part
            spliting x into x_female and x_male depending on the genders
        '''
        # 根据gender的取值将x分成x_f和x_m
        x_f = x[genders.squeeze() == 0]  # 选择gender为0的行作为女性数据
        x_m = x[genders.squeeze() == 1]  # 选择gender为1的行作为男性数据

        mesh_f, joints_f = self.smpl_f(
            pose=x_f[:, 10:82].clone(),
            betas=x_f[:, :10].clone(),
            trans=x_f[:, 82:85].clone()
        )

        mesh_m, joints_m = self.smpl_m(
            pose=x_m[:, 10:82].clone(),
            betas=x_m[:, :10].clone(),
            trans=x_m[:, 82:85].clone()
        )

        batch_size = x.size(0)

        processed_mesh = torch.empty((batch_size, 6890, 3)).to(self.device)  # 创建与x相同大小的空张量
        processed_mesh[genders.squeeze() == 0] = mesh_f  # 将x_f数据放回原始位置
        processed_mesh[genders.squeeze() == 1] = mesh_m  # 将x_m数据放回原始位置

        processed_joints = torch.empty((batch_size, 24, 3)).to(self.device)  # 创建与x相同大小的空张量
        processed_joints[genders.squeeze() == 0] = joints_f  # 将x_f数据放回原始位置
        processed_joints[genders.squeeze() == 1] = joints_m  # 将x_m数据放回原始位置

        return processed_mesh, processed_joints.reshape(-1,72)

def mobilenet(device):
    # 创建 MobileNetV2 模型
    return MobileNetV2(num_classes=10+72+3, device = device)
    
# # 打印模型结构
# print(model)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    pose_size = 72
    beta_size = 10

    np.random.seed(9608)
    batch_size = 64
    pose = torch.from_numpy((np.random.rand(batch_size, pose_size) - 0.5) * 0.4)\
            .type(torch.float32).to(device)
    betas = torch.from_numpy((np.random.rand(batch_size, beta_size) - 0.5) * 0.06) \
            .type(torch.float32).to(device)
    trans = torch.from_numpy(np.ones((batch_size, 3))).type(torch.float32).to(device)
    print(pose.shape,betas.shape,trans.shape)
    image = torch.rand((batch_size, 1, 224, 224), dtype=torch.float32).to(device)
    # print(image)
    model = mobilenet(device).to(device)

    mesh, batch = model(image)
    print(mesh.shape,batch.shape)




    # meshs,joints = test_smpl(device=device, pose = batch[:,10:82].clone().detach(), betas = batch[:,:10].clone().detach(), trans = batch[:,82:85].clone().detach() )
     
    # mesh, joint = test_smpl(device,pose,betas,trans)