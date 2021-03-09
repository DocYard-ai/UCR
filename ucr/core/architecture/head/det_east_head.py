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
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            act=None,
            groups=1):
        super(ConvBNLayer, self).__init__()
        
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


class EASTHead(nn.Module):
    """
    """
    def __init__(self, in_channels, model_name, **kwargs):
        super(EASTHead, self).__init__()
        self.model_name = model_name
        if self.model_name == "large":
            num_outputs = [128, 64, 1, 8]
        else:
            num_outputs = [64, 32, 1, 8]

        self.det_conv1 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=num_outputs[0],
            kernel_size=3,
            stride=1,
            padding=1,
            act='relu')
        self.det_conv2 = ConvBNLayer(
            in_channels=num_outputs[0],
            out_channels=num_outputs[1],
            kernel_size=3,
            stride=1,
            padding=1,
            act='relu')
        self.score_conv = ConvBNLayer(
            in_channels=num_outputs[1],
            out_channels=num_outputs[2],
            kernel_size=1,
            stride=1,
            padding=0,
            act=None)
        self.geo_conv = ConvBNLayer(
            in_channels=num_outputs[1],
            out_channels=num_outputs[3],
            kernel_size=1,
            stride=1,
            padding=0,
            act=None)

    def forward(self, x):
        f_det = self.det_conv1(x)
        f_det = self.det_conv2(f_det)
        f_score = self.score_conv(f_det)
        f_score = torch.sigmoid(f_score)
        f_geo = self.geo_conv(f_det)
        f_geo = (torch.sigmoid(f_geo) - 0.5) * 2 * 800

        pred = {'f_score': f_score, 'f_geo': f_geo}
        return pred
