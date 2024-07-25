# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 23:04:59 2023

@author: Administrator
"""
import os
import torch
from PIL import Image
from tqdm import tqdm
import numpy as np

from torch.optim import Adam,SGD
import logging
from network.models.custom_losses import CustomSquareLoss
from network.utils.visualization import CIR_Single_Point,Building_CIR_Integration,Building_HeatMap,split_image

def train(configs,model,train_loader,test_loader,val_loader=None):

    num_epochs=configs.epochs
    if configs.is_ddp:
        length=min(len(train_loader),int(configs.epoch_data_num//(configs.world_size*configs.batch_size)))
    else:
        length=min(len(train_loader),int(configs.epoch_data_num//configs.batch_size))
        
    if not os.path.exists(configs.log_dir):
        os.makedirs(configs.log_dir)

    logging.basicConfig(filename=f"{configs.log_dir}/{configs.log_name}", level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    
    #Init
    if not (configs.train_load_name is None):
        model.load(configs.train_load_name)

    best_loss = float('inf')

    if configs.output_channel==1:
        init_optimizer = SGD(model.init_params, lr=0.005)
        for epoch in range(20):
            if configs.is_ddp:
                train_loader.sampler.set_epoch(epoch)
            #train
            model.network.train()
            train_loss = 0.0
            length=min(len(train_loader),length)
            if (not configs.is_ddp) or (configs.is_ddp and configs.rank==0):
                t=tqdm(train_loader, desc=f"Epoch Init {epoch+1}", total=length)
            else:
                t=train_loader

            for batch_idx, batch in enumerate(t):
                if batch_idx==length:
                    break
                sample,target,data_name = batch
                loss=model.train(sample, target,init_optimizer)
                train_loss += loss.item()
                lr=init_optimizer.param_groups[0]['lr']
                if (not configs.is_ddp) or (configs.is_ddp and configs.rank==0):
                    t.set_postfix({"epoch":epoch+1, "batch":batch_idx+1, "loss":loss.item(),"lr":lr})

            
            average_loss = train_loss / length
            if average_loss < best_loss:
                best_loss = average_loss
                if (not configs.is_ddp) or (configs.is_ddp and configs.rank==0):
                    model.save("model_init")
            if (not configs.is_ddp) or (configs.is_ddp and configs.rank==0):
                logging.info(f'Epoch [{epoch+1}/20] Loss: {average_loss:.4f}')
            
        logging.shutdown()
        return

    loss_indicate=configs.train_output_channel
    if configs.train_loss_equalize:
        loss_indicate_list=[1/(loss_indicate-i) for i in range(loss_indicate)]+[0 for i in range(configs.output_channel-loss_indicate)]
    else:
        loss_indicate_list=[1 for i in range(loss_indicate)]+[0 for i in range(configs.output_channel-loss_indicate)]

    #Start Train
    for epoch in range(num_epochs):
        if configs.is_ddp:
            train_loader.sampler.set_epoch(epoch)
        #train
        model.network.train()
        train_loss = 0.0
        
        constants = torch.tensor(loss_indicate_list).to(configs.device)
        
        constants = constants.view(1, len(constants), 1, 1)
        
        if (not configs.is_ddp) or (configs.is_ddp and configs.rank==0):
            t=tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", total=length)
        else:
            t=train_loader

        for batch_idx, batch in enumerate(t):
            if batch_idx==length:
                break
            sample,target,data_name = batch
            loss=model.train(sample, target,loss_function_parm=constants)
            train_loss += loss.item()
            lr=model.optimizer.param_groups[0]['lr']
            if (not configs.is_ddp) or (configs.is_ddp and configs.rank==0):
                t.set_postfix({"epoch":epoch+1, "batch":batch_idx+1,"loss_indicate":loss_indicate, "loss":loss.item(),"lr":lr})
            model.scheduler_CyclicLR.step()

        model.optimizer.param_groups[0]['lr']=model.scheduler_CyclicLR.base_lrs[0]
        #val
        model.network.train()
        if not (val_loader is None):
            val_loss = 0.0
            if (not configs.is_ddp) or (configs.is_ddp and configs.rank==0):
                t=tqdm(val_loader, desc=f"Epoch val {epoch+1}/{num_epochs}", total=len(val_loader),leave=False)
            else:
                t=val_loader

            with torch.no_grad():
                for batch_idx, batch in enumerate(t):
                    sample,target,data_name = batch
                    loss_val=model.val(sample, target,loss_function_parm=constants)
                    val_loss += loss_val.item()
                    if (not configs.is_ddp) or (configs.is_ddp and configs.rank==0):
                        t.set_postfix({"epoch":epoch+1, "batch":batch_idx+1, "loss":loss_val.item()})
                        
                 
        average_train_loss = train_loss / length
        print(f"Epoch {epoch+1}/{num_epochs}, Average train Loss: {average_train_loss:.4f}")
        
        
        if not (val_loader is None):
            average_loss=val_loss / len(val_loader)
            if (not configs.is_ddp) or (configs.is_ddp and configs.rank==0):
                print(f"Epoch {epoch+1}/{num_epochs}, Average val Loss: {average_loss:.4f}")
            if (not configs.is_ddp) or (configs.is_ddp and configs.rank==0):
                logging.info(f'Epoch [{epoch+1}/{num_epochs}] Loss: {average_train_loss:.4f} Val: {average_loss:.4f} Indicate: {loss_indicate}')
            
            loss_=loss_val
        else:
            average_loss=average_train_loss
            if (not configs.is_ddp) or (configs.is_ddp and configs.rank==0):
                logging.info(f'Epoch [{epoch+1}/{num_epochs}] Loss: {average_train_loss:.4f} Indicate: {loss_indicate}')
            loss_=loss
        model.scheduler.step(loss_)
        if average_loss < best_loss:
            best_loss = average_loss
            if (not configs.is_ddp) or (configs.is_ddp and configs.rank==0):
                model.save(configs.save_name)

        model.scheduler_CyclicLR.base_lrs = [model.optimizer.param_groups[0]['lr']]
        model.scheduler_CyclicLR.max_lrs = [model.optimizer.param_groups[0]['lr']*10]
            
        torch.cuda.empty_cache()
    if (not configs.is_ddp) or (configs.is_ddp and configs.rank==0):
        model.save("model_finish")
    logging.shutdown()
    return


def test(configs,model,test_loader,test_load_name=None,is_Integration=False,is_Heatmap=False,is_split=False):
    if not os.path.exists(configs.output_dir):
        os.makedirs(configs.output_dir)
    if test_load_name is None:
        model.load(configs.test_load_name)
    else:
        model.load(test_load_name)
    model.network.train()
    
    loss_indicate=configs.train_output_channel
    loss_indicate_list=[1 for i in range(loss_indicate)]+[0 for i in range(configs.output_channel-loss_indicate)]
    constants = torch.tensor(loss_indicate_list).to(configs.device)
    constants = constants.view(1, len(constants), 1, 1)
    
    test_loss = 0.0
    
    length=min(len(test_loader),configs.test_data_num//configs.batch_size_test)
    with tqdm(test_loader, desc="Test",total=length,leave=False) as t:
        for batch_idx, batch in enumerate(t):
            if batch_idx==length:
                break
            sample,target,data_name = batch
            output,loss=model.test(sample, target,loss_function_parm=constants)
            test_loss+=loss.item()
            for i in range(output.shape[0]):
                save_path=f"{configs.output_dir}/{data_name[i].split('/')[-1]}"
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                for j in range(output.shape[1]):
                    image = Image.fromarray(output[i,j,:,:].astype(np.uint8), 'L')
                    image.save(f"{save_path}/{str(j).zfill(8)}.png")
            t.set_postfix({"batch":batch_idx+1, "loss":loss.item()})
        average_test_loss = test_loss / length
        print(f"Average test Loss: {average_test_loss:.4f}")
    

    # #Building and CIR integration
    if is_Integration:
        Building_CIR_Integration(f"{configs.dataset_path}/{configs.dataset_input}",f"{configs.output_dir}",f"{configs.output_integration_dir}",alpha=0.5,regeneration=True)
    
    if is_Heatmap:
        Building_HeatMap(f"{configs.output_dir}",f"{configs.output_heatmap_dir}",regeneration=True)

    if is_split:
        split_image(f"{configs.output_dir}",f"{configs.output_split_dir}",regeneration=True)
    
    return






