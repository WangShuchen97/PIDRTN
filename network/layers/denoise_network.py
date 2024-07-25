# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 16:18:25 2023

@author: Administrator

"""

import torch
import torch.nn as nn
from network.layers.basic import BasicConv, Down, Up, Outc
class Denoise_Network(nn.Module):
    def __init__(self,in_channels, out_channels,base_c=32):
        super(Denoise_Network, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.base_c=base_c
        
        self.inc = BasicConv(in_channels, base_c * 1, mid_channels=base_c)
        self.down1 = Down(base_c * 1, base_c * 2)     
        self.down2 = Down(base_c * 2, base_c * 4)     
        self.down3 = Down(base_c * 4, base_c * 8)     
        self.down4 = Down(base_c * 8, base_c * 16)     
        self.down5 = Down(base_c * 16, base_c * 32)     

        self.up1 = Up(base_c * 32, base_c * 16)
        self.up2 = Up(base_c * 16, base_c * 8)
        self.up3 = Up(base_c * 8, base_c * 4)
        self.up4 = Up(base_c * 4, base_c * 2)
        self.up5 = Up(base_c * 2, base_c * 1)
        
        self.outc = Outc(base_c * 1, out_channels,mid_channels=base_c)


    def forward(self, x):
        x0 = self.inc(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        
        y = self.up1(x5, x4)
        y = self.up2(y, x3)
        y = self.up3(y, x2)
        y = self.up4(y, x1)
        y = self.up5(y, x0)

        y=self.outc(y)
        
        return y
