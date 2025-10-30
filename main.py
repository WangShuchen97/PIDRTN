import os
import torch
import argparse
import numpy as np
import time
import torchvision.transforms as transforms
import torch.distributed as dist
import sys

from network.data_provider import datasets_factory
from network.models import model_factory
from network import trainer

import warnings
warnings.filterwarnings("ignore")

# os.environ['CUDA_VISIBLE_DEVICES']='2'

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='PIDRTN')
parser.add_argument('--device', type=str, default='cuda:0',help="If not ddp")
parser.add_argument('--cpu_worker', type=int, default=4,help="how many subprocesses to use for data loading")
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--model_name', type=str, default='PIDRTN')
#data configs
parser.add_argument('--dataset_name', type=str, default='raytracing')
parser.add_argument('--dataset_path', type=str,default='data')
parser.add_argument('--dataset_test_path', type=str,default='data')
parser.add_argument('--dataset_input', type=str,default='input')
parser.add_argument('--dataset_output', type=str,default='output_overlap_32')
parser.add_argument('--is_overlap', type=int,default=1)
parser.add_argument('--dataset_size', type=int, default=5000)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--batch_size_test', type=int, default=5)
parser.add_argument('--batch_size_val', type=int, default=5)
parser.add_argument('--input_channel', type=int, default=1)
parser.add_argument('--input_height', type=int, default=512)
parser.add_argument('--input_width', type=int, default=512)
parser.add_argument('--output_channel', type=int, default=32)
parser.add_argument('--output_times', type=int, default=2.5)
parser.add_argument('--test_ratio', type=float, default=0.2,help="Divide the test data from all data")
parser.add_argument('--val_ratio', type=float, default=0.05,help="Divide the validation data from the training data")
parser.add_argument('--input_mean', type=list, default=[0.5],help="[] indicates no normalization")
parser.add_argument('--input_std', type=list, default=[0.5],help="[] indicates no normalization")
parser.add_argument('--p_RandomHorizontalFlip', type=float, default=0.5)
parser.add_argument('--p_RandomVerticalFlip', type=float, default=0.5)
parser.add_argument('--p_RandomRotate', type=float, default=0)#Deprecated
#train configs
parser.add_argument('--mode', type=str,default="test")
parser.add_argument('--is_ddp', type=int,default=0)
parser.add_argument('--is_apex', type=int,default=0)
parser.add_argument('--rank', type=int,default=0)
parser.add_argument('--world_size', type=int,default=4)
parser.add_argument('--loss_function', type=str,default="CustomSquareLoss")
parser.add_argument('--ref_test_size', type=int,default=12)
parser.add_argument('--checkpoint_path', type=str,default='results/checkpoints_ddp')
parser.add_argument('--train_load_name', type=str,default=None)
parser.add_argument('--test_load_name', type=str,default='model_best')
parser.add_argument('--save_name', type=str,default='model_best')
parser.add_argument('--loss_weight', type=float,default=1)
parser.add_argument('--loss2_weight', type=float,default=0)
parser.add_argument('--train_output_channel', type=int,default=32)
parser.add_argument('--train_loss_equalize', type=int,default=0)
parser.add_argument('--epochs', type=int,default=40)
parser.add_argument('--epoch_data_num', type=int,default=1280)
parser.add_argument('--test_data_num', type=int,default=495)
parser.add_argument('--l2_weight_decay', type=float,default=0.01)
parser.add_argument('--learn_rate', type=float,default=0.0001)
parser.add_argument('--learn_rate_patience', type=int,default=2)
parser.add_argument('--learn_rate_factor', type=float,default=0.5)
parser.add_argument('--learn_rate_min', type=float,default=1e-10)
parser.add_argument('--learn_cooldown', type=int,default=2)
parser.add_argument('--learn_step_size_up', type=int,default=20)
parser.add_argument('--learn_threshold', type=int,default=0.001)
parser.add_argument('--learn_threshold_mode', type=str,default="abs")
parser.add_argument('--max_grad_norm', type=float,default=100)
parser.add_argument('--log_dir', type=str,default='results/log')
parser.add_argument('--log_name', type=str,default='Training.log')
parser.add_argument('--output_dir', type=str, default='results/results_output', help='Path to save generated images')
parser.add_argument('--output_integration_dir', type=str, default='results/results_integration', help='Path to save generated images')
parser.add_argument('--output_heatmap_dir', type=str, default='results/results_heatmap', help='Path to save generated images')
parser.add_argument('--output_split_dir', type=str, default='results/results_split', help='Path to save generated images')


try:
    configs = parser.parse_args()
except:
    configs = parser.parse_args(args=[])


transform=None
# transform = transforms.Compose([
#     transforms.CenterCrop(200),
#     transforms.Resize((128,128))
# ])

torch.manual_seed(configs.seed)


def main(configs,is_training=True,is_validation=False):
    
    if configs.is_ddp:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        configs.rank=rank
        print(f"Start running basic DDP example on rank {rank}.")
        configs.device = "cuda:"+str(rank % torch.cuda.device_count())
        
    model =model_factory.Model(configs)
    
    try:
        if not configs.is_ddp:
            model.net_structure(mode="torchsummary")#torchviz or torchsummary
    except:
        pass
    
    if is_training:
        if is_validation:
            train_loader,val_loader,test_loader = datasets_factory.data_provider(configs,transform=transform,is_training=True,is_validation=True)
        else:
            train_loader,test_loader = datasets_factory.data_provider(configs,transform=transform,is_training=True,is_validation=False)
    else:
        test_loader = datasets_factory.data_provider(configs,transform=transform,is_training=False)

    if configs.mode=="test":   
        trainer.test(configs,model,test_loader)
    if configs.mode=="train":
        if is_validation:
            trainer.train(configs,model,train_loader,test_loader,val_loader)
        else:
            trainer.train(configs,model,train_loader,test_loader)
    if configs.is_ddp:
        dist.destroy_process_group()
    time.asctime()


    
if __name__ == "__main__":
      
    # configs.model_name='PIDRTN'
    # configs.model_name='PIDRTN-A'
    # configs.model_name='U-net'
    
    if configs.model_name=='PIDRTN-A':
        configs.dataset_name='raytracing_ref'
        if configs.mode=="test":        
            configs.ref_x=np.linspace(0, configs.input_height-1, configs.ref_test_size+2, dtype=int).tolist()[1:-1]
            configs.ref_y=np.linspace(0, configs.input_width-1,  configs.ref_test_size+2, dtype=int).tolist()[1:-1]
        else:
            configs.ref_x=np.linspace(0, configs.input_height-1, 12+2, dtype=int).tolist()[1:-1]
            configs.ref_y=np.linspace(0, configs.input_width-1, 12+2, dtype=int).tolist()[1:-1]
    if configs.mode=="test":
        if configs.model_name=='PIDRTN':
            configs.test_load_name='PIDRTN'
            configs.output_dir="results/PIDRTN"
        if configs.model_name=='PIDRTN-A':
            configs.test_load_name='PIDRTN-A'
            configs.output_dir="results/PIDRTN-A"
        if configs.model_name=='U-net':
            configs.test_load_name='U-net'
            configs.output_dir="results/U-net"
        main(configs,is_training=False)
    if configs.mode=="train":
        if configs.is_ddp==0:
            configs.checkpoint_path='results/checkpoints'
            configs.output_channel=32
            configs.train_output_channel=32
            configs.world_size=1
            main(configs,is_training=True,is_validation=True)   
        else:
            configs.checkpoint_path='results/checkpoints_ddp'
            configs.output_channel=32
            configs.train_output_channel=32
            main(configs,is_training=True,is_validation=True)   
    

            

        
