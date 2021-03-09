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

import torch
from torch import nn
import torch.nn.functional as F


class ConvBNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 groups=1,
                 if_act=True,
                 act=None,
                 name=None):
        super(ConvBNLayer, self).__init__()
        self.if_act = if_act
        self.act = act
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False)
  
        self.bn = nn.BatchNorm2d(
            out_channels)
        
        if self.act=='relu':
            self.relu=nn.ReLU(inplace=True)

    def forward(self, inputs):
        y = self.conv(inputs)
        y = self.bn(y)
        if self.act=='relu':
            y = self.relu(y)
        return y
    

class DeConvBNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 groups=1,
                 if_act=True,
                 act=None,
                 name=None):
        super(DeConvBNLayer, self).__init__()
        
        self.act = act
        self.deconv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False)

        self.bn = nn.BatchNorm2d(
            out_channels,
            )
        if self.act=='relu':
            self.relu=nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.deconv(x)
        x = self.bn(x)
        if self.act=='relu':
            x = self.relu(x)
        return x


class EASTFPN(nn.Module):
    def __init__(self, in_channels, model_name, **kwargs):
        super(EASTFPN, self).__init__()
        self.model_name = model_name
        if self.model_name == "large":
            self.out_channels = 128
        else:
            self.out_channels = 64
        self.in_channels = in_channels[::-1]
        self.h1_conv = ConvBNLayer(
            in_channels=self.out_channels+self.in_channels[1],
            out_channels=self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            act='relu')
        self.h2_conv = ConvBNLayer(
            in_channels=self.out_channels+self.in_channels[2],
            out_channels=self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            act='relu')
        self.h3_conv = ConvBNLayer(
            in_channels=self.out_channels+self.in_channels[3],
            out_channels=self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            act='relu')
        self.g0_deconv = DeConvBNLayer(
            in_channels=self.in_channels[0],
            out_channels=self.out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
            act='relu')
        self.g1_deconv = DeConvBNLayer(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
            act='relu')
        self.g2_deconv = DeConvBNLayer(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
            act='relu')
        self.g3_conv = ConvBNLayer(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            act='relu')

    def forward(self, x):
        f = x[::-1]

        h = f[0]
        g = self.g0_deconv(h)
        h = torch.cat((g, f[1]), dim=1)
        h = self.h1_conv(h)
        g = self.g1_deconv(h)
        h = torch.cat((g, f[2]), dim=1)
        h = self.h2_conv(h)
        g = self.g2_deconv(h)
        h = torch.cat((g, f[3]), dim=1)
        h = self.h3_conv(h)
        g = self.g3_conv(h)

        return g