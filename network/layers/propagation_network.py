# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 18:20:31 2023

@author: Administrator
"""
import torch
import torch.nn as nn
from network.layers.basic import BasicConv, InceptionBlock, Down, Up, Outc

class Propagation_Network(nn.Module):
    def __init__(self,in_channels=32, out_channels=32, base_c=32):
        super(Propagation_Network, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        base_c = base_c

        self.inc = InceptionBlock(1,8,4,8,4,8,8)
        
        self.inc2 = BasicConv(base_c * 1, base_c * 1) 

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
        self.outc = Outc(base_c * 1, out_channels, mid_channels = base_c, kernel = 3)

        self.up1_vx = Up(base_c * 32, base_c * 16)
        self.up2_vx = Up(base_c * 16, base_c * 8)
        self.up3_vx = Up(base_c * 8, base_c * 4)
        self.up4_vx = Up(base_c * 4, base_c * 2)
        self.up5_vx = Up(base_c * 2, base_c * 1)
        self.outc_vx = Outc(base_c * 1, out_channels * 1, mid_channels = base_c, kernel = 3)

        self.up1_vy = Up(base_c * 32, base_c * 16)
        self.up2_vy = Up(base_c * 16, base_c * 8)
        self.up3_vy = Up(base_c * 8, base_c * 4)
        self.up4_vy = Up(base_c * 4, base_c * 2)
        self.up5_vy = Up(base_c * 2, base_c * 1)
        self.outc_vy = Outc(base_c * 1, out_channels * 1, mid_channels = base_c, kernel = 3)

    def forward(self, x):
        
        x = self.inc(x)
        x1=self.inc2(x)

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)

        z = self.up1(x6, x5)
        z = self.up2(z, x4)
        z = self.up3(z, x3)
        z = self.up4(z, x2)
        z = self.up5(z, x1)
        z = self.outc(z)

        vx = self.up1_vx(x6, x5)
        vx = self.up2_vx(vx, x4)
        vx = self.up3_vx(vx, x3)
        vx = self.up4_vx(vx, x2)
        vx = self.up5_vx(vx, x1)
        vx = self.outc_vx(vx)

        vy = self.up1_vy(x6, x5)
        vy = self.up2_vy(vy, x4)
        vy = self.up3_vy(vy, x3)
        vy = self.up4_vy(vy, x2)
        vy = self.up5_vy(vy, x1)
        vy = self.outc_vy(vy)

        return z, vx, vy




