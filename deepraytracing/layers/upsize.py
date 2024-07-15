# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 16:18:25 2023

@author: Administrator

"""

import torch.nn as nn

class Upsize(nn.Module):
    def __init__(self,channel):
        super(Upsize, self).__init__()

        self.up_1 = nn.ConvTranspose2d(channel, channel, kernel_size=2, stride=1)
        self.up_2 = nn.ConvTranspose2d(channel, channel, kernel_size=3, stride=1)
        self.up_3 = nn.ConvTranspose2d(channel, channel, kernel_size=3, stride=1)
        self.up_4 = nn.ConvTranspose2d(channel, channel, kernel_size=3, stride=1)
        self.up_5 = nn.ConvTranspose2d(channel, channel, kernel_size=2, stride=2)
        self.up_6 = nn.ConvTranspose2d(channel, channel, kernel_size=2, stride=2)

    def forward(self, s):
        
        s = self.up_1(s)
        s = self.up_2(s)
        s = self.up_3(s)
        s = self.up_4(s)
        s = self.up_5(s)
        s = self.up_6(s)
        
        return s
