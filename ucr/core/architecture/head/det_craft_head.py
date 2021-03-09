from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.init as init

class CRAFTHead(nn.Module):
    def __init__(self, in_channels):
        super(CRAFTHead, self).__init__()

        num_class = 2
        self.conv_cls = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, num_class, kernel_size=1),
        )

        init_weights(self.conv_cls.modules())
        
    def forward(self, x):
        
        y = self.conv_cls(x)
        
        text_map = y[:,0,:,:]
        score_map = y[:,1,:,:]
        
        pred = {'text_map': text_map, 'score_map': score_map}
        return pred


def init_weights(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()