# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 17:59:53 2023

@author: Administrator
"""

import torch.nn as nn
import torch

class CustomSquareLoss(nn.Module):
    def __init__(self,loss_weight,loss2_weight):
        super(CustomSquareLoss, self).__init__()
        
        self.loss_weight=loss_weight
        self.loss2_weight=loss2_weight

    def forward(self, input, target,constants=None):

        if constants is None:
            loss = torch.mean((input - target)**2)
        else:
            
            loss=(input - target)**2
            
            loss=loss * constants
            
            loss=torch.mean(loss)
            
            for i in range(input.shape[1]):
                if i==0:
                    loss2=constants[:,i,:,:]*(input[:,i,:,:] - target[:,i,:,:])**2
                else:
                    loss2=loss2+constants[:,i,:,:]*(input[:,i,:,:]-input[:,i-1,:,:] - target[:,i,:,:]+ target[:,i-1,:,:])**2
            loss2=loss2/input.shape[1]
            loss2=torch.mean(loss2)
            loss=loss*self.loss_weight+loss2*self.loss2_weight
                
        return loss


class CustomSplitLoss(nn.Module):
    def __init__(self,loss_weight,loss2_weight,num=10):
        super(CustomSplitLoss, self).__init__()
        self.loss_weight=loss_weight
        self.loss2_weight=loss2_weight
        self.num=num

    def forward(self, input, target,constants=None):
        
        for i in range(input.shape[1]):
            if i>=input.shape[1]-1-self.num and input.shape[1]-1-self.num>0:
                if i==input.shape[1]-1-self.num:
                    loss2=constants[:,i,:,:]*(input[:,i,:,:]-input[:,i-1,:,:] - target[:,i,:,:]+ target[:,i-1,:,:])**2
                else:
                    loss2=loss2+constants[:,i,:,:]*(input[:,i,:,:]-input[:,i-1,:,:] - target[:,i,:,:]+ target[:,i-1,:,:])**2
            if i==0:
                loss=(input[:,i,:,:] - target[:,i,:,:])**2
            else:
                loss=loss+(input[:,i,:,:]-input[:,i-1,:,:] - target[:,i,:,:]+ target[:,i-1,:,:])**2
        loss=loss/input.shape[1]
        loss=torch.mean(loss)
        if input.shape[1]-1-self.num>0:
            loss2=loss2/self.num
            loss2=torch.mean(loss2)
            loss=loss*self.loss_weight+loss2*self.loss2_weight
        return loss
                