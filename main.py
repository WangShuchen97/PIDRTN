import os
import numpy as np
import torch
import argparse
import time
import torchvision.transforms as transforms
import torch.distributed as dist
import warnings
import datetime

warnings.filterwarnings("ignore")

from network.data_provider import datasets_factory
from network.models import model_factory
from network import trainer
from network.utils.tool import make_dir


# os.environ['CUDA_VISIBLE_DEVICES']='2'

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

parser = argparse.ArgumentParser(description='RT')
parser.add_argument('--model_name', type=str, default='RT')
parser.add_argument('--data_provider', type=str, default='pt_pt')
parser.add_argument('--mode', type=str,default="train")
parser.add_argument('--loss_function', type=str,default="CustomSquareLoss")
parser.add_argument('--device', type=str, default='cuda:0',help="If not ddp")
parser.add_argument('--cpu_worker', type=int, default=4,help="how many subprocesses to use for data loading")
parser.add_argument('--world_size', type=int,default=4)
parser.add_argument('--timestamp', type=str, default=timestamp)
parser.add_argument('--is_amp', type=int,default=0)
#ddp configs
parser.add_argument('--is_ddp', type=int,default=0)
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--rank', type=int,default=0)
#data configs
parser.add_argument('--dataset_input', type=str,default='./data/input')
parser.add_argument('--dataset_output', type=str,default='./data/output_rt')
parser.add_argument('--maximum_sample_size', type=int, default=5000)
parser.add_argument('--test_ratio', type=float, default=0.2,help="Divide the test data from all data")
parser.add_argument('--is_val', type=int, default=1)
parser.add_argument('--val_ratio', type=float, default=0.1,help="Divide the validation data from the training data")
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--batch_size_test', type=int, default=2)
parser.add_argument('--batch_size_val', type=int, default=2)
#train configs
parser.add_argument('--checkpoint_path', type=str,default='results/checkpoints')
parser.add_argument('--train_load_name', type=str,default=None)
parser.add_argument('--test_load_name', type=str,default='model_best')
parser.add_argument('--save_name', type=str,default='model_best')
parser.add_argument('--save_name_final', type=str,default='model_Final')
parser.add_argument('--learn_rate', type=float,default=0.00005)
parser.add_argument('--learn_rate_patience', type=int,default=2)
parser.add_argument('--learn_rate_factor', type=float,default=0.5)
parser.add_argument('--learn_rate_min', type=float,default=1e-10)
parser.add_argument('--learn_cooldown', type=int,default=2)
parser.add_argument('--learn_step_size_up', type=int,default=20)
parser.add_argument('--learn_threshold', type=int,default=0.001)
parser.add_argument('--learn_threshold_mode', type=str,default="abs")
parser.add_argument('--max_grad_norm', type=float,default=100)
parser.add_argument('--output_dir', type=str, default='results/results_output', help='Path to save generated images')
parser.add_argument('--log_dir', type=str,default='results/log')
parser.add_argument('--epochs', type=int,default=200)
parser.add_argument('--epoch_data_num', type=int,default=2000)
parser.add_argument('--test_data_num', type=int,default=495)
parser.add_argument('--l2_weight_decay', type=float,default=0.01)


#optional
parser.add_argument('--train_channel', type=int,default=2)
parser.add_argument('--model_mode', type=str, default="RayMap", help="All, RayMap, main")
parser.add_argument('--input_mean', type=list, default=[],help="[] indicates no normalization")
parser.add_argument('--input_std', type=list, default=[],help="[] indicates no normalization")
parser.add_argument('--output_times', type=float, default=1)
parser.add_argument('--p_RandomHorizontalFlip', type=float, default=0.5)
parser.add_argument('--p_RandomVerticalFlip', type=float, default=0.5)
parser.add_argument('--p_RandomRotate', type=float, default=0)#Deprecated


try:
    configs = parser.parse_args()
except:
    configs = parser.parse_args(args=[])

torch.manual_seed(configs.seed)



def main(configs):
    
    if configs.is_ddp:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        configs.rank=rank
        print(f"Start running basic DDP example on rank {rank}.")
        configs.device = "cuda:"+str(rank % torch.cuda.device_count())
    
    train_loader,val_loader,test_loader = datasets_factory.data_provider(configs,mode="train")
        
    input_example, target_example, data_name_example = next(iter(test_loader))
    print("Input shape:", list(input_example.shape))
    print("Output shape:", list(target_example.shape))
    configs.input_shape=list(input_example.shape)
    configs.output_shape=list(target_example.shape)

    model =model_factory.Model(configs)
    
    
    try:
        if not configs.is_ddp:
            model.net_structure(input_size=configs.input_shape,mode="torchsummary")#torchviz or torchsummary
    except:
        pass
    
    if configs.mode=="train":
        trainer.train(configs,model,train_loader,test_loader,val_loader)
    if configs.mode=="test":   
        trainer.test(configs,model,test_loader)
        
    if configs.is_ddp:
        dist.destroy_process_group()
    time.asctime()

   
if __name__ == "__main__":

    main(configs)   

        

