import torch

import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F

import sys
import math


class Bottleneck(nn.Module):
    def __init__(self, num_channels, growth_rate):
        super().__init__()
        inter_channels = 4 * growth_rate
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv1 = nn.Conv2d(
            num_channels, inter_channels, kernel_size=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(inter_channels)
        self.conv2 = nn.Conv2d(
            inter_channels, growth_rate, kernel_size=3, padding=1, bias=False
        )

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat((x, out), 1)
        return out


class SingleLayer(nn.Module):
    def __init__(self, num_channels, growth_rate):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv1 = nn.Conv2d(
            num_channels, growth_rate, kernel_size=3, padding=1, bias=False
        )

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = torch.cat((x, out), 1)
        return out


class Transition(nn.Module):
    def __init__(self, num_channels, num_out_channels):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv1 = nn.Conv2d(
            num_channels, num_out_channels, kernel_size=1, bias=False
        )

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):
    def __init__(self, growth_rate, depth, reduction, num_classes, bottleneck):
        super().__init__()

        num_dense_blocks = (depth - 4) // 3
        if bottleneck:
            num_dense_blocks //= 2

        num_channels = 2 * growth_rate
        self.conv1 = nn.Conv2d(
            3, num_channels, kernel_size=3, padding=1, bias=False
        )
        self.dense1 = self._make_dense(
            num_channels, growth_rate, num_dense_blocks, bottleneck
        )
        num_channels += num_dense_blocks * growth_rate
        num_out_channels = int(math.floor(num_channels * reduction))
        self.trans1 = Transition(num_channels, num_out_channels)

        num_channels = num_out_channels
        self.dense2 = self._make_dense(
            num_channels, growth_rate, num_dense_blocks, bottleneck
        )
        num_channels += num_dense_blocks * growth_rate
        num_out_channels = int(math.floor(num_channels * reduction))
        self.trans2 = Transition(num_channels, num_out_channels)

        num_channels = num_out_channels
        self.dense3 = self._make_dense(
            num_channels, growth_rate, num_dense_blocks, bottleneck
        )
        num_channels += num_dense_blocks * growth_rate

        self.bn1 = nn.BatchNorm2d(num_channels)
        self.fc = nn.Linear(num_channels, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_dense(
        self, num_channels, growth_rate, num_dense_blocks, bottleneck
    ):
        layers = []
        for i in range(int(num_dense_blocks)):
            if bottleneck:
                layers.append(Bottleneck(num_channels, growth_rate))
            else:
                layers.append(SingleLayer(num_channels, growth_rate))
            num_channels += growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.dense3(out)
        out = torch.squeeze(F.avg_pool2d(F.relu(self.bn1(out)), 8))
        out = F.log_softmax(self.fc(out), dim=1)
        return out
