# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 22:53:19 2023

@author: Administrator
"""
import torch
import torch.nn as nn

from network.layers.init_bias import Init_Bias
from network.layers.propagation_network import Propagation_Network
from network.layers.basic import Outc
from network.layers.denoise_network import Denoise_Network
from network.layers.fusion_network import Fusion_Network_2
from network.utils.tool import warp, make_grid

class PIDRTN_A(nn.Module):
    def __init__(self,configs):
        super(PIDRTN_A, self).__init__()
        
        self.configs = configs
        self.height = configs.input_height
        self.width = configs.input_width
        self.channel=configs.input_channel
        self.pred_length=configs.output_channel
        
        self.Init_Bias = Init_Bias(self.height,self.width)
        
        if self.configs.output_channel>1:
            
            self.Propagation_Network=Propagation_Network(self.configs.input_channel,self.pred_length,base_c=32)

            self.grid = make_grid(1, self.configs.input_channel, self.configs.input_height, self.configs.input_width,self.configs.device)

            self.Outc=Outc(configs.output_channel,configs.output_channel,configs.output_channel)
            
            self.Denoise_Network=Denoise_Network(self.pred_length,self.pred_length,base_c=32)
            
            self.Fusion_Layer_list=[]
            for i in range(self.pred_length):
                self.Fusion_Layer_list.append(Fusion_Network_2(2,1).to(configs.device))
            
    def forward(self, x_list):
        
        x=x_list[0]
        s=x_list[1]
        
        if x.device.type!=self.configs.device:
            x=x.to(self.configs.device)
        if s.device.type!=self.configs.device:
            s=s.to(self.configs.device)
        
        batch = x.shape[0]

        frames = self.Init_Bias(x)
        
        if self.pred_length==1:
            return frames

        else:
            series=[]
            
            self.grid=self.grid.to(frames.dtype)
            
            grid = self.grid.repeat(batch, 1, 1, 1)
            
            intensity,motion_x,motion_y=self.Propagation_Network(x)
            
            intensity = intensity.reshape(batch, self.pred_length, 1, self.height, self.width)

            motion_x = motion_x.reshape(batch, self.pred_length, 1, self.height, self.width)

            motion_y = motion_y.reshape(batch, self.pred_length, 1, self.height, self.width)

            motion = torch.cat([motion_x,motion_y],dim=2)
            
            for i in range(self.pred_length):
                
                frames_ = warp(frames, motion[:,i], grid, mode="bilinear", padding_mode="zeros")
                frames_ = frames_ + 20*intensity[:,i]-10
                
                if self.configs.is_overlap:
                    frames=frames_+frames
                else:
                    frames=frames_
            
                frames= self.Fusion_Layer_list[i](frames,s[:,i:i+1,:,:])
                
                series.append(frames)
                
            y = torch.cat(series, dim=1)
 
            y=self.Outc(y)

            y=self.Denoise_Network(y)
            
            y = torch.clamp(y, min=0,max=255)
            
            return y
        
            



    
        
        