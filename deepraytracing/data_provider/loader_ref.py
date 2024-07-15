# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 22:43:52 2023

@author: Administrator
"""

import os
import cv2
import numpy as np
import random
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF


class InputHandle(Dataset):
    def __init__(self,input_param):
        self.data=input_param["data"]
        self.targets=input_param["targets"]
        self.transform = input_param["transform"]
        self.output_channel=input_param["output_channel"]
        self.input_data_type=input_param["input_data_type"]
        self.output_data_type=input_param["output_data_type"]
        
        self.input_mean=input_param["input_mean"]
        self.input_std=input_param["input_std"]
        self.p_RandomHorizontalFlip=input_param["p_RandomHorizontalFlip"]
        self.p_RandomVerticalFlip=input_param["p_RandomVerticalFlip"]
        self.p_RandomRotate=input_param["p_RandomRotate"]
        self.configs=input_param["configs"]
        random.seed(input_param["seed"])
        
        self.ref_x={6: np.linspace(0, self.configs.input_height-1, 6+2, dtype=int).tolist()[1:-1],
                    8: np.linspace(0, self.configs.input_height-1, 8+2, dtype=int).tolist()[1:-1],
                    10:np.linspace(0, self.configs.input_height-1, 10+2, dtype=int).tolist()[1:-1],
                    12:np.linspace(0, self.configs.input_height-1, 12+2, dtype=int).tolist()[1:-1],
                    14:np.linspace(0, self.configs.input_height-1, 14+2, dtype=int).tolist()[1:-1],
                    16:np.linspace(0, self.configs.input_height-1, 16+2, dtype=int).tolist()[1:-1]}
        self.ref_y={6: np.linspace(0, self.configs.input_width-1, 6+2, dtype=int).tolist()[1:-1],
                    8: np.linspace(0, self.configs.input_width-1, 8+2, dtype=int).tolist()[1:-1],
                    10:np.linspace(0, self.configs.input_width-1, 10+2, dtype=int).tolist()[1:-1],
                    12:np.linspace(0, self.configs.input_width-1, 12+2, dtype=int).tolist()[1:-1],
                    14:np.linspace(0, self.configs.input_width-1, 14+2, dtype=int).tolist()[1:-1],
                    16:np.linspace(0, self.configs.input_width-1, 16+2, dtype=int).tolist()[1:-1]}
        
    def random_transform_together(self):

        if self.p_RandomHorizontalFlip>random.random():
            pp_RandomHorizontalFlip=1
        else:
            pp_RandomHorizontalFlip=0
        if self.p_RandomVerticalFlip>random.random():
            pp_RandomVerticalFlip=1
        else:
            pp_RandomVerticalFlip=0
                   
        temp=[transforms.RandomHorizontalFlip(pp_RandomHorizontalFlip),                                                                                                
              transforms.RandomVerticalFlip(pp_RandomVerticalFlip) ]
        if self.p_RandomRotate>random.random():
            pp_RandomRotate=random.randint(0,360)
            RandomRotate = lambda x: TF.rotate(x, pp_RandomRotate)
            temp.append(transforms.Lambda(RandomRotate))
            
        transform = transforms.Compose(temp)            
        return transform

    def load(self, index):
        #sample (1,weight,height)
        #target (time,weight,height)
        
        data_name=self.targets[index]
        
        transform_together=self.random_transform_together()
        sample = cv2.imread(self.data[index], 2)
        sample=sample.astype(self.input_data_type)
      
        sample=transforms.ToTensor()(sample)
        
        if len(self.input_mean)>0 and len(self.input_mean)==len(self.input_std):
            sample=transforms.Normalize(self.input_mean,self.input_std)(sample)
        
        if not (self.transform is None):
            sample = self.transform(sample)
        sample = transform_together(sample)
        target=[]

        target_list=os.listdir(self.targets[index])
        target_list.sort()

        length=min(len(target_list),self.output_channel)
        for i in range(length):
            img_path=self.targets[index]+'/'+target_list[i]
            img = cv2.imread(img_path, 2)
            img = img.astype(self.output_data_type)
            img=transforms.ToTensor()(img)
            if not (self.transform is None):
                img = self.transform(img)
            img = transform_together(img)
            target.append(img)

        target = np.concatenate(target, axis=0)
        target=torch.FloatTensor(target)
        target=target*self.configs.output_times
        #============================
        if self.configs.ref_x is None:
            indicate=random.choice([6,8,10,12,14,16])
            ref_x=self.ref_x[indicate]
            ref_y=self.ref_y[indicate]
        else:
            ref_x=self.configs.ref_x
            ref_y=self.configs.ref_y
            
        sample2 = target[:, ref_x,:][:, :, ref_y].detach().clone()
        sample2 = torch.nn.functional.interpolate(sample2.unsqueeze(0), size=(self.configs.input_height,self.configs.input_width), mode='bilinear', align_corners=False)
        
        sample2 = sample2.squeeze()
        
        sample2 = sample2.detach().clone()
        
        return [sample,sample2],target,data_name
    
    def __getitem__(self, index):
        sample, target, data_name=self.load(index)
        return sample, target, data_name
    
    def __len__(self):
        return len(self.data)
        
