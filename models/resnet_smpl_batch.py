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
from smpl.smpl_torch_batch import SMPLModel

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
        self.name = 'resnet18_smpl'
        self.device = device

        # 初始化对应性别的SMPL模型  
        model_path_f = r'D:\workspace\python_ws\pose-master\smpl\basicModel_f_lbs_10_207_0_v1.0.0.pkl'
        self.smpl_f = SMPLModel(device=self.device,model_path=model_path_f).to(device)

        model_path_m = r'D:\workspace\python_ws\pose-master\smpl\basicmodel_m_lbs_10_207_0_v1.0.0.pkl'
        self.smpl_m = SMPLModel(device=self.device,model_path=model_path_m).to(device)


    def make_layer(self, block, out_channels, blocks, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x, genders):
        # print(genders[0])
        '''
        skeleton: torch.Size([batchsize, 72])
        image: torch.Size([batchsize, 1, 224, 224])
        gender: torch.Size([batchsize, 1])
        trans: torch.Size([batchsize, 3])
        
        '''

        '''
        全连接层输出长度为85的X
        x[0:10]: betas
        x[10:82]: pose 
        x[82:85]: trans
        ----------
        x[10:12]描述了根节点的rotation
        x[82:85]描述了更节点的global translation
        '''
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



def posenet(num_classes=10+72+3,device='cuda'):#默认直接预测出24×3的关节点位置

    return ResNet_smpl(BasicBlock, [2, 2, 2, 2], num_classes,device)



if __name__ == "__main__":
# 创建ResNet-18模型
    # 创建ResNet模型实例
    
    model = posenet(10+72) # 10个shape参数和 24*3的pose参数

    # smpl = SMPLModel()
    # 打印模型结构
    print(model)
