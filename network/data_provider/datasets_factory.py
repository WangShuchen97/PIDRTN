# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 23:09:38 2023

@author: Administrator
"""

from network.data_provider import loader
from network.data_provider import loader_ref
from torch.utils.data import DataLoader
import os
import torch
from sklearn.model_selection import train_test_split
import random


datasets_map = {
    'raytracing': loader,
    'raytracing_ref': loader_ref
}

def data_provider(configs,transform=None,is_training=True,is_validation=True):
    if configs.dataset_name not in datasets_map:
        raise ValueError('Name of dataset unknown %s' % configs.dataset_name)
    if configs.dataset_name == 'raytracing' or configs.dataset_name == 'raytracing_ref' :
        if is_training:
            dataset_path=configs.dataset_path
        else:
            dataset_path=configs.dataset_test_path
        name_list = os.listdir(dataset_path+"/"+configs.dataset_input)
        case_list=[]
        target_list=[]
        
        num=0        
        for name in name_list:
            case_list.append(dataset_path + '/'+configs.dataset_input+'/' + name)
            target_list.append(dataset_path + '/'+configs.dataset_output+'/data' + name[3:-4])
            num+=1
            if num >=configs.dataset_size:
                break
        
        if is_training:
            train_data, test_data, train_targets, test_targets = train_test_split(case_list, target_list, test_size=configs.test_ratio, random_state=configs.seed)
            if is_validation:
                train_data, val_data, train_targets, val_targets = train_test_split(train_data, train_targets, test_size=configs.val_ratio, random_state=configs.seed)
        else:
            if configs.test_data_num==495 and len(target_list)==4977:
                #To ensure that the results of each test are the same, take the following samples for testing (the random numbers for Linux and Windows may sometimes be different)
                teatdata_num=['1023', '1028', '1061', '106', '108', '1140', '1156', '1159', '1173', '1187', '1193', '1201', '1218', '123', '1243', '1255', '1286', '1294', '1308', '1312', '1321', '1340', '1429', '1484', '1491', '1495', '1496', '1509', '1539', '1545', '1565', '1576', '1594', '159', '1600', '1615', '162', '1663', '1687', '1728', '1731', '1734', '1736', '1752', '1765', '1784', '178', '1796', '1801', '1803', '1807', '1822', '1839', '1842', '1844', '184', '1851', '1872', '1876', '1877', '1912', '1927', '1930', '1940', '1959', '1974', '1979', '1986', '198', '2033', '2038', '2055', '2084', '2102', '2113', '215', '216', '2193', '2194', '2212', '2213', '2230', '2235', '2244', '2246', '2249', '2264', '2301', '2324', '2333', '2334', '2355', '2367', '2385', '2432', '2433', '2434', '2442', '245', '2464', '2465', '2469', '2480', '2491', '2535', '2537', '2558', '255', '2564', '2575', '2591', '2632', '2641', '2646', '2649', '2653', '2660', '2665', '2674', '2683', '2714', '271', '2731', '2753', '2755', '2768', '2769', '2773', '2779', '278', '2794', '2823', '2853', '2855', '2867', '2878', '2889', '2899', '2912', '2915', '2926', '2945', '2962', '2964', '3007', '3014', '3018', '303', '3042', '304', '3079', '307', '3085', '3111', '3131', '3135', '3154', '3159', '315', '3221', '3236', '329', '3306', '3342', '3366', '3384', '338', '3410', '3435', '3439', '3443', '3459', '3460', '3500', '3507', '350', '3532', '3535', '3541', '3550', '3556', '3581', '3605', '3615', '3625', '3636', '3657', '3677', '3685', '3699', '3706', '3720', '3761', '3777', '3778', '3780', '3787', '3807', '3817', '3818', '3823', '3829', '382', '3841', '3866', '3867', '389', '38', '3918', '3930', '3931', '3960', '3976', '3986', '3994', '4023', '4069', '4076', '4085', '4094', '4118', '4132', '4151', '4152', '4172', '422', '4250', '4257', '4279', '4282', '4306', '4309', '4342', '4372', '43', '4407', '4421', '4422', '442', '443', '4441', '4451', '4457', '4521', '4568', '4580', '4596', '4598', '4617', '4647', '4661', '466', '4699', '4704', '4714', '4718', '4755', '4757', '4759', '4783', '4797', '4816', '4819', '4823', '4845', '4851', '4856', '4901', '4911', '4949', '4957', '4958', '497', '498', '5000', '5009', '5010', '5025', '5058', '5075', '5087', '5098', '5106', '5118', '5131', '5187', '518', '519', '5202', '5236', '523', '5272', '527', '5296', '5336', '5359', '5361', '5363', '5373', '5379', '5389', '5393', '5405', '5409', '5419', '5429', '5439', '5449', '5464', '5481', '5492', '5496', '549', '5506', '5529', '5556', '5557', '5564', '5604', '5609', '5620', '563', '564', '5659', '5666', '5672', '5676', '5677', '5679', '5691', '5696', '5699', '5702', '5708', '5724', '5731', '5734', '5752', '5777', '5802', '5803', '5833', '5844', '5850', '5857', '5861', '5865', '5877', '5880', '5889', '5890', '5927', '5940', '594', '5963', '5981', '5986', '5987', '5995', '6014', '6028', '6035', '6037', '6042', '6064', '6066', '6084', '6105', '6115', '6132', '6133', '6164', '6180', '6202', '6222', '6238', '6261', '6265', '6289', '6290', '6292', '6308', '6343', '6360', '6426', '6470', '6488', '6494', '6495', '6497', '64', '6509', '6510', '6539', '6540', '6544', '654', '6558', '6584', '659', '6605', '660', '6613', '6628', '6636', '6663', '6696', '6700', '6737', '673', '6756', '6766', '676', '678', '6795', '6824', '6828', '683', '6844', '6846', '6852', '6862', '6944', '7032', '7048', '7052', '7094', '7127', '7177', '7188', '718', '7205', '7209', '7213', '7217', '7222', '7247', '7258', '7265', '726', '7281', '7286', '7313', '7321', '7337', '7351', '7356', '7379', '7392', '7405', '7422', '7516', '7570', '7581', '7588', '7626', '7636', '7663', '7666', '7674', '7703', '7730', '7742', '7750', '7766', '776', '7770', '7773', '7790', '7797', '783', '785', '7861', '7890', '7900', '7917', '7938', '7940', '7966', '7970', '7981', '810', '811', '813', '828', '82', '83', '872', '881', '888', '8', '90', '914', '928', '998']
                
                test_targets=[]
                test_data=[]
                for i in range(len(name_list)):
                    k=0
                    for j in range(len(name_list[i])):
                        if name_list[i][j]=="_":
                            k+=1
                            if k==2:
                                break
                    if name_list[i][4:j] in teatdata_num:
                        test_targets.append(target_list[i])
                        test_data.append(case_list[i])
            else:
                random.seed(configs.seed)
                random_list = list(zip(case_list, target_list))
                random.shuffle(random_list) 
                test_data, test_targets = zip(*random_list)  

        test_input_param = {"data":test_data,
                            "targets":test_targets,
                            "output_channel":configs.output_channel,
                            "input_data_type":'float32',
                            "output_data_type":'float32',
                            "transform":transform,
                            "input_mean":configs.input_mean,
                            "input_std":configs.input_std,
                            "p_RandomHorizontalFlip":0,
                            "p_RandomVerticalFlip":0,
                            "p_RandomRotate":0,
                            "seed":configs.seed,
                            "configs":configs
                            }
        test_input_handle = datasets_map[configs.dataset_name].InputHandle(test_input_param)
        test_input_handle = DataLoader(test_input_handle,
                                       batch_size=configs.batch_size_val,
                                       shuffle=False,
                                       num_workers=configs.cpu_worker,
                                       drop_last=False)
        if is_training:
            train_input_param = {"data":train_data,
                                "targets":train_targets,
                                "output_channel":configs.output_channel,
                                "input_data_type":'float32',
                                "output_data_type":'float32',
                                "transform":transform,
                                "input_mean":configs.input_mean,
                                "input_std":configs.input_std,
                                "p_RandomHorizontalFlip":configs.p_RandomHorizontalFlip,
                                "p_RandomVerticalFlip":configs.p_RandomVerticalFlip,
                                "p_RandomRotate":configs.p_RandomRotate,
                                "seed":configs.seed,
                                "configs":configs
                                }
            train_input_handle = datasets_map[configs.dataset_name].InputHandle(train_input_param)
            if configs.is_ddp:
                sampler = torch.utils.data.distributed.DistributedSampler(train_input_handle, num_replicas=configs.world_size, rank=configs.rank)
                train_input_handle = DataLoader(train_input_handle,
                                               batch_size=configs.batch_size,
                                               num_workers=configs.cpu_worker,
                                               drop_last=True,
                                               sampler=sampler,
                                               pin_memory=True)
            else:    
                train_input_handle = DataLoader(train_input_handle,
                                               batch_size=configs.batch_size,
                                               shuffle=True,
                                               num_workers=configs.cpu_worker,
                                               drop_last=True)
            if is_validation:
                val_input_param = {"data":val_data,
                                    "targets":val_targets,
                                    "output_channel":configs.output_channel,
                                    "input_data_type":'float32',
                                    "output_data_type":'float32',
                                    "transform":transform,
                                    "input_mean":configs.input_mean,
                                    "input_std":configs.input_std,
                                    "p_RandomHorizontalFlip":0,
                                    "p_RandomVerticalFlip":0,
                                    "p_RandomRotate":0,
                                    "seed":configs.seed,
                                    "configs":configs
                                    }
                val_input_handle = datasets_map[configs.dataset_name].InputHandle(val_input_param)
                val_input_handle = DataLoader(val_input_handle,
                                               batch_size=configs.batch_size_val,
                                               shuffle=False,
                                               num_workers=configs.cpu_worker,
                                               drop_last=False)
                return train_input_handle,val_input_handle,test_input_handle
            else:
                return train_input_handle,test_input_handle
        return test_input_handle

