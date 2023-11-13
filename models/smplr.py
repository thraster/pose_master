import torch
import torch.nn as nn

class MLPWithBranch(nn.Module):
    def __init__(self, input_sizes, output_size):
        '''
        scaler = 2: 整体变窄2倍
        
        '''
        super(MLPWithBranch, self).__init__()
        # 定义输入大小和输出大小
        self.input_size = [input_sizes, 1024, 1024, 512, 512, 512]
        self.input_sizes_main = [512, 256, 256, output_size]
        self.input_sizes_branch = [512, 256, output_size]
        self.output_size = output_size

        self.fc_last = nn.Linear(output_size*2, output_size)

        # 主分支的层
        self.layers_main = nn.ModuleList()
        for i in range(len(self.input_size) - 1):
            linear_layer = nn.Linear(self.input_size[i], self.input_size[i+1])
            bn_layer = nn.BatchNorm1d(self.input_size[i+1])
            relu_layer = nn.ReLU()

            self.layers_main.append(linear_layer)
            self.layers_main.append(bn_layer)
            self.layers_main.append(relu_layer)


        self.output_layer_main = nn.ModuleList()
        for i in range(len(self.input_sizes_main) - 1):
            linear_layer = nn.Linear(self.input_sizes_main[i], self.input_sizes_main[i+1])
            bn_layer = nn.BatchNorm1d(self.input_sizes_main[i+1])
            relu_layer = nn.ReLU()

            self.output_layer_main.append(linear_layer)
            self.output_layer_main.append(bn_layer)
            self.output_layer_main.append(relu_layer)

        self.output_layer_branch = nn.ModuleList()
        for i in range(len(self.input_sizes_branch) - 1):
            linear_layer = nn.Linear(self.input_sizes_branch[i], self.input_sizes_branch[i+1])
            bn_layer = nn.BatchNorm1d(self.input_sizes_branch[i+1])
            relu_layer = nn.ReLU()

            self.output_layer_branch.append(linear_layer)
            self.output_layer_branch.append(bn_layer)
            self.output_layer_branch.append(relu_layer)


    def forward(self, x):
        # 主分支
        x_main = x
        for layer in self.layers_main:
            x_main = layer(x_main)

        y1 = y2 = x_main

        for layer in self.output_layer_main:
            y1 = layer(y1)

        for layer in self.output_layer_branch:
            y2 = layer(y2)

        
        # x_main = self.output_layer_main(x_main)

        # # 分支
        # x_branch = self.output_layer_branch(x)

        # 进行 concatenation
        # concatenated_output = torch.cat((y1, y2), dim=1)
        # output = self.fc_last(concatenated_output.clone())

        return y1+y2


class SMLPR_MLP(nn.Module):
    def __init__(self, device = 'cuda'):
        super(SMLPR_MLP, self).__init__()
        self.shape_encoder = MLPWithBranch(73, 10).to(device)
        self.pose_encoder = MLPWithBranch(73, 72).to(device)
        self.name = 'SMLPR_MLP_layer'

    def forward(self, x):
        shape = self.shape_encoder(x)
        pose = self.pose_encoder(x)
        return shape, pose
        

if __name__ == '__main__':
    input_sizes = 72
    output_sizes = 10
    scaler = 1
    # # 创建带有分支的 MLP 模型
    # mlp_with_branch_model = MLPWithBranch(input_sizes, output_sizes)

    # # # 打印模型结构
    # # print(mlp_with_branch_model)
    # # print(mlp_with_branch_model.output_layer_main)
    # # print(mlp_with_branch_model.output_layer_branch)
    # y = mlp_with_branch_model(torch.rand(8,72))
    # print(y.shape)
    # print(y)


    model = SMLPR_MLP()
    print(model.name)
    print(model)