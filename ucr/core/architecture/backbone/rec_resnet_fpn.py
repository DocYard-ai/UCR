# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Modifications copyright (c) 2021 DocYard Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch import nn
import torch
import numpy as np

__all__ = ["ResNetFPN"]


class ResNetFPN(nn.Module):
    def __init__(self, in_channels=1, layers=50, **kwargs):
        super(ResNetFPN, self).__init__()
        supported_layers = {
            18: {
                'depth': [2, 2, 2, 2],
                'block_class': BasicBlock
            },
            34: {
                'depth': [3, 4, 6, 3],
                'block_class': BasicBlock
            },
            50: {
                'depth': [3, 4, 6, 3],
                'block_class': BottleneckBlock
            },
            101: {
                'depth': [3, 4, 23, 3],
                'block_class': BottleneckBlock
            },
            152: {
                'depth': [3, 8, 36, 3],
                'block_class': BottleneckBlock
            }
        }
        stride_list = [(2, 2), (2, 2), (1, 1), (1, 1)]
        num_filters = [64, 128, 256, 512]
        self.depth = supported_layers[layers]['depth']
        self.conv = ConvBNLayer(
            in_channels=in_channels,
            out_channels=64,
            kernel_size=7,
            stride=2,
            act="relu")
        self.block_list = []
        in_ch = 64
        if layers >= 50:
            for block in range(len(self.depth)):
                for i in range(self.depth[block]):
                    blocks = BottleneckBlock(
                            in_channels=in_ch,
                            out_channels=num_filters[block],
                            stride=stride_list[block] if i == 0 else 1)
                    self.add_module(
                        "bottleneckBlock_{}_{}".format(block, i),
                        blocks)
                    in_ch = num_filters[block] * 4
                    self.block_list.append(blocks)
        else:
            for block in range(len(self.depth)):
                for i in range(self.depth[block]):
                    conv_ = "res" + str(block + 2) + chr(97 + i)
                    if i == 0 and block != 0:
                        stride = (2, 1)
                    else:
                        stride = (1, 1)
                    basic_block = BasicBlock(
                            in_channels=in_ch,
                            out_channels=num_filters[block],
                            stride=stride_list[block] if i == 0 else 1,
                            is_first=block == i == 0)
                    self.add_module(
                        conv_,
                        basic_block)
                    in_ch = basic_block.out_channels
                    self.block_list.append(basic_block)
        out_ch_list = [in_ch // 4, in_ch // 2, in_ch]
        self.base_block = []
        self.conv_trans = []
        self.bn_block = []
        for i in [-2, -3]:
            in_channels = out_ch_list[i + 1] + out_ch_list[i]
            block1 = nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_ch_list[i],
                        kernel_size=1)
            self.add_module(
                    "F_{}_base_block_0".format(i),
                    block1)
            self.base_block.append(
                block1)
            
            block2 = nn.Conv2d(
                        in_channels=out_ch_list[i],
                        out_channels=out_ch_list[i],
                        kernel_size=3,
                        padding=1)
            self.add_module(
                    "F_{}_base_block_1".format(i),
                    block2)
            self.base_block.append(
                block2)
            
            block3 = nn.BatchNorm2d(out_ch_list[i])
            self.add_module(
                    "F_{}_base_block_2".format(i),
                    block3)
            self.base_block.append(
                block3)
        
        block4 = nn.Conv2d(
                    in_channels=out_ch_list[i],
                    out_channels=512,
                    kernel_size=1)
        self.add_module(
                "F_{}_base_block_3".format(i),
                block4)
        self.base_block.append(
            block4)
        self.out_channels = 512

    def forward(self, x):
        x = self.conv(x)
        fpn_list = []
        F = []
        for i in range(len(self.depth)):
            fpn_list.append(np.sum(self.depth[:i + 1]))

        for i, block in enumerate(self.block_list):
            x = block(x)
            for number in fpn_list:
                if i + 1 == number:
                    F.append(x)
        base = F[-1]

        j = 0
        for i, block in enumerate(self.base_block):
            
            if i % 3 == 0:
                base = nn.functional.relu(base)
            if i % 3 == 0 and i < 6:
                j = j + 1
                b, c, w, h = F[-j - 1].shape
                if [w, h] == list(base.shape[2:]):
                    base = base
                else:
                    base = self.conv_trans[j - 1](base)
                    base = self.bn_block[j - 1](base)
                base = torch.cat((base, F[-j - 1]), dim=1)
            base = block(base)
        return base
    

class ConvBNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 groups=1,
                 act=None):
        
        self.act = act
        super(ConvBNLayer, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=2 if stride == (1, 1) else kernel_size,	
            dilation=2 if stride == (1, 1) else 1,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            groups=groups,
            bias=False)

        self.bn = nn.BatchNorm2d(out_channels)
        
        if self.act=='relu':
            self.relu=nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.act=='relu':
            x = self.relu(x)
        return x


class ShortCut(nn.Module):
    def __init__(self, in_channels, out_channels, stride, is_first=False):
        super(ShortCut, self).__init__()
        self.use_conv = True

        if in_channels != out_channels or stride != 1 or is_first == True:
            if stride == (1, 1):
                self.conv = ConvBNLayer(
                    in_channels, out_channels, 1, 1)
            else:  # stride==(2,2)
                self.conv = ConvBNLayer(
                    in_channels, out_channels, 1, stride)
        else:
            self.use_conv = False

    def forward(self, x):
        if self.use_conv:
            x = self.conv(x)
        return x


class BottleneckBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride):
        super(BottleneckBlock, self).__init__()

        self.conv0 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            act='relu')
        
        self.conv1 = ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            act='relu')
        
        self.conv2 = ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels * 4,
            kernel_size=1,
            act=None)
        
        self.short = ShortCut(
            in_channels=in_channels,
            out_channels=out_channels * 4,
            stride=stride,
            is_first=False)
        
        self.out_channels = out_channels * 4
        
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, inputs):
        y = self.conv0(inputs)
        y = self.relu1(y)        
        conv1 = self.conv1(y)
        conv1 = self.relu2(conv1)
        conv2 = self.conv2(conv1)
        short = self.short(inputs)
        y = torch.add(short, conv2)
        y = self.relu3(y)
        return y


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, is_first):
        super(BasicBlock, self).__init__()
        self.conv0 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            act='relu',
            stride=stride)
        self.conv1 = ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            act=None)
        self.short = ShortCut(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            is_first=is_first)
        
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

        self.out_channels = out_channels

    def forward(self, x):
        y = self.conv0(x)
        y = self.relu1(y)        
        y = self.conv1(y)
        y = y + self.short(x)
        return self.relu2(y)
