# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 23:10:24 2023

@author: Administrator
"""


import os
import torch
import torch.nn as nn
from torch.optim import Adam,SGD
from torchviz import make_dot
from torchsummary import summary
from torch.optim.lr_scheduler import ReduceLROnPlateau,CyclicLR
from network.models import PIDRTN
from network.models import PIDRTN_A
from network.models import U_net
from network.models.custom_losses import CustomSquareLoss,CustomSplitLoss
from torch.nn.utils import clip_grad_norm_
from torch.nn.parallel import DistributedDataParallel as DDP
#from apex import amp

class Model(object):
    def __init__(self, configs):
        self.configs = configs
        networks_map = {
            'PIDRTN': PIDRTN.PIDRTN,
            'PIDRTN-A': PIDRTN_A.PIDRTN_A,
            'U-net': U_net.U_net
        }
        loss_map={
            'CrossEntropyLoss':nn.CrossEntropyLoss(),
            'MSELoss':nn.MSELoss(),
            'CustomSquareLoss':CustomSquareLoss(configs.loss_weight,configs.loss2_weight),
            'CustomSplitLoss':CustomSplitLoss(configs.loss_weight,configs.loss2_weight)
            }
        if configs.model_name in networks_map:
            Network = networks_map[configs.model_name]
            self.network = Network(configs).to(configs.device)
            
            try:
                self.init_params=list(self.network.Init_Bias.parameters())
            except:
                pass
            
            self.loss_function=loss_map[configs.loss_function]
            
            self.optimizer = SGD(self.network.parameters(), lr=configs.learn_rate,weight_decay=configs.l2_weight_decay)
            
            if configs.is_apex:
                self.network, self.optimizer = amp.initialize(self.network, self.optimizer, opt_level='O1',max_loss_scale=1)
            if configs.is_ddp:
                #self.network = DDP(self.network, device_ids=[int(self.configs.device[-1])],find_unused_parameters=True)
                self.network = DDP(self.network, device_ids=[int(self.configs.device[-1])])
                
            
            
            self.scheduler_CyclicLR = CyclicLR(self.optimizer, 
                                              base_lr=configs.learn_rate, 
                                              max_lr=configs.learn_rate*10, 
                                              step_size_up=configs.learn_step_size_up,
                                              mode='exp_range')
            
            self.scheduler = ReduceLROnPlateau(self.optimizer, 
                                               mode='min', 
                                               patience=configs.learn_rate_patience, 
                                               factor=configs.learn_rate_factor, 
                                               verbose=True,
                                               eps=0,
                                               cooldown=configs.learn_cooldown,
                                               min_lr=configs.learn_rate_min,
                                               threshold=configs.learn_threshold,
                                               threshold_mode=configs.learn_threshold_mode)
        else:
            raise ValueError('Name of network unknown %s' % configs.model_name)
    def net_structure(self,mode=None):
        if mode=="torchviz":
            x = torch.randn(self.configs.batch_size, self.configs.input_channel, self.configs.input_height, self.configs.input_width).to(self.configs.device)
            y = self.network(x)
            make_dot(y, params=dict(list(self.network.named_parameters()))).render("SimpleNet", format="png")
        if mode=="torchsummary":
            summary(self.network, input_size=(self.configs.input_channel,self.configs.input_height,self.configs.input_width), batch_size=-1, device="cpu")
        return
         
    def to_device(self,x):
        if type(x) == list:
            for i in range(len(x)):
                x[i]=torch.FloatTensor(x[i]).to(self.configs.device)
        else:
            x=torch.FloatTensor(x).to(self.configs.device)
        return x
        
    def train(self, sample, target,optimizer=None,loss_function_parm=None):
        
        sample=self.to_device(sample)
        target=self.to_device(target)
        
        output=self.network(sample)
        if loss_function_parm is None:
            loss = self.loss_function(output, target)  
        else:
            loss = self.loss_function(output, target,loss_function_parm) 
        if optimizer is None:
            optimizer=self.optimizer
            
        optimizer.zero_grad()  # 清零梯度
        if self.configs.is_apex:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()  # 反向传播
        
        if not (self.configs.max_grad_norm is None):
            clip_grad_norm_(self.network.parameters(), self.configs.max_grad_norm)

        optimizer.step()  # 更新模型参数
            
        return loss
    
    def val(self, sample, target,loss_function_parm=None):
        
        sample=self.to_device(sample)
        target=self.to_device(target)
        
        output=self.network(sample)
        if loss_function_parm is None:
            loss = self.loss_function(output, target)  
        else:
            loss = self.loss_function(output, target,loss_function_parm)  
        return loss
    
    def test(self, sample, target,loss_function_parm=None):
        
        sample=self.to_device(sample)
        target=self.to_device(target)
        
        with torch.no_grad():
            output=self.network(sample)
        if loss_function_parm is None:
            loss = self.loss_function(output, target)  
        else:
            loss = self.loss_function(output, target,loss_function_parm) 
        return output.detach().cpu().numpy(),loss.detach().cpu().numpy()


    def save(self, save_name="model"):
        
        if self.configs.is_ddp:
            if self.configs.rank!=0:
                return
            
        stats = {'model_state_dict': self.network.state_dict(),
                 'optimizer_state_dict': self.optimizer.state_dict(),}
        
        checkpoint_path = f"{self.configs.checkpoint_path}/{save_name}.pth"
        
        if not os.path.exists(self.configs.checkpoint_path):
            os.makedirs(self.configs.checkpoint_path)
        
        torch.save(stats, checkpoint_path)
        print("save model to %s" % checkpoint_path)
        return
        
    def load(self,load_name="model",load_param=None):
        if load_name is None:
            return
        checkpoint_path = f"{self.configs.checkpoint_path}/{load_name}.pth"
        
        if os.path.exists(checkpoint_path):
            stats = torch.load(checkpoint_path)
            
            #Ensure that the model and saved layer names are consistent for DDP and non parallel scenarios
            is_module_in_pth=False
            for key, _ in stats['model_state_dict'].items():
                if key.startswith('module.'):
                    is_module_in_pth=True
                    break
            is_module_in_model=False
            for key, _ in self.network.state_dict().items():
                if key.startswith('module.'):
                    is_module_in_pth=True
                    break
            if is_module_in_pth:
                if is_module_in_model:
                    pass
                else:
                    stats['model_state_dict'] = {k.replace('module.', ''): v for k, v in stats['model_state_dict'].items()}
            else:
                if is_module_in_model:
                    stats['model_state_dict'] = {'module.'+k: v for k, v in stats['model_state_dict'].items()}
                else:
                    pass
            #===============================================

            if load_param is None:
                if self.configs.is_ddp:
                    self.network.load_state_dict(stats['model_state_dict'],strict=False)
                else:
                    self.network.load_state_dict(stats['model_state_dict'],strict=False)
            else:
                new_model_dict = self.network.state_dict()
                for param_name in load_param:
                    print(param_name)
                    new_model_dict[param_name] = stats['model_state_dict'][param_name]
                if self.configs.is_ddp:
                    self.network.load_state_dict(new_model_dict)
                else:
                    self.network.load_state_dict(new_model_dict)
            if (not self.configs.rank) or (self.configs.is_ddp and self.configs.rank==0):
                print("load model from %s" % checkpoint_path)
        else:
            print(f"The checkpoint file '{checkpoint_path}' does not exist.")
        return
    
    


    



    
