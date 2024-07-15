# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 16:18:25 2023

@author: Administrator

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Fusion_Network(nn.Module):
    def __init__(self,in_channels, out_channels):
        super(Fusion_Network, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.inc=BasicConv(in_channels,out_channels)
        
        self.conv1=BasicConv(out_channels,out_channels)
        self.conv2=BasicConv(out_channels,out_channels)
        self.conv3=BasicConv(out_channels,out_channels)
    def forward(self, x):
        
        x = self.inc(x)
       
        x = self.conv1(x)
        
        x = self.conv2(x)
        
        x = self.conv3(x)
        
        return x
    
class Fusion_Network_2(nn.Module):
    def __init__(self,in_channels, out_channels):
        super(Fusion_Network_2, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.inc=BasicConv(in_channels,in_channels)
        
        self.conv1=BasicConv(in_channels,out_channels)

    def forward(self, x,s):
        
        y = torch.cat([x,s],dim=1)
        
        y = self.inc(y)
       
        y = self.conv1(y)
        
        return y


class BasicConv(nn.Module):

    def   __init__(self, in_channels, out_channels, kernel=3, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, mid_channels, kernel_size=kernel, padding="same"),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=kernel, padding="same"),
        )
    def forward(self, x):
        x = self.conv(x)
        return x





