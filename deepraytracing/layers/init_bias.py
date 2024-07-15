# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 22:21:47 2023

@author: Administrator
"""

import torch
import torch.nn as nn


import torch.nn.init as init

class Init_Bias(nn.Module):
    def __init__(self,H,W):
        super(Init_Bias, self).__init__()
        self.H=H
        self.W=W
        self.bias = nn.Parameter(torch.rand(1,H,W))
        
        init.uniform_(self.bias) 
        
    def forward(self, x):

        batch=x.size(0)
        y=self.bias.repeat(batch, 1, 1,1)
        y=y*1000
        return y
    

