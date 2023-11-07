import torch
import torch.nn as nn
import sys
import pickle
import numpy as np
import cv2
import time

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

    def forward(self, x, genders):
        meshs = []
        joints = []
        x = self.features(x)
        '''
        全连接层输出长度为85的X
        x[0:10]: betas
        x[10:82]: pose 
        x[82:85]: trans
        ----------
        x[10:12]描述了根节点的rotation
        x[82:85]描述了更节点的global translation
        '''

        x = self.classifier(x)
        for i, batch in enumerate(x):
            # print(batch.shape)
            # print(i)
            if genders[i,0].int() == 0:
                mesh, joint = self.forward_smpl_f(betas = batch[:10], pose = batch[10:82],trans = batch[82:85])
            elif genders[i,0].int() == 1:
                mesh, joint = self.forward_smpl_m(betas = batch[:10], pose = batch[10:82],trans = batch[82:85])
            meshs.append(mesh)
            joints.append(torch.reshape(joint,(1,72)))
        # print(f"smpl forward time: {smpl2-smpl1}")
        meshs = torch.cat(meshs, dim = 0)
        joints = torch.cat(joints, dim = 0)

        return meshs, joints

def mobilenet(device):
    # 创建 MobileNetV2 模型
    return MobileNetV2(num_classes=10+72+3, device = device)
    
# # 打印模型结构
# print(model)
