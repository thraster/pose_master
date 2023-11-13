import torch
import torch.nn as nn
from torchsummary import summary

class HourglassBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks = 1):
        super(HourglassBlock, self).__init__()

        self.up1 = self.conv_block(in_channels, out_channels)

        self.down1 = nn.MaxPool2d(2, 2)
        self.low1 = self.conv_block(out_channels, out_channels)
        if num_blocks > 1:
            self.low2 = HourglassBlock(out_channels, out_channels, num_blocks - 1)
        else:
            self.low2 = None
        self.low3 = self.conv_block(out_channels, out_channels)

        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        up1 = self.up1(x)
        down1 = self.down1(x)
        low1 = self.low1(down1)
        if self.low2 is not None:
            low2 = self.low2(low1)
        else:
            low2 = low1
        low3 = self.low3(low2)
        up2 = self.up2(low3)

        return up1 + up2

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

class StackedHourglassNet(nn.Module):
    def __init__(self, num_stacks, num_blocks, num_classes):
        super(StackedHourglassNet, self).__init__()

        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        self.hourglass_blocks = self.make_hourglass_blocks(num_blocks, num_stacks)

        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pre(x)

        for hourglass_block in self.hourglass_blocks:
            x = hourglass_block(x)

        x = self.fc(x)

        return x

    def make_hourglass_blocks(self, num_blocks, num_stacks):
        blocks = []
        for _ in range(num_stacks):
            for _ in range(num_blocks):
                blocks.append(HourglassBlock(64, 64))

        return nn.Sequential(*blocks)

# 创建模型
num_stacks = 2
num_blocks = 4
num_classes = 16
model = StackedHourglassNet(num_stacks, num_blocks, num_classes).to('cuda')

# 打印模型结构
print(model)
summary(model, input_size=(3, 224, 224))