# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 16:56:56 2023

@author: Administrator
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from network.utils.visualization import CIR_Single_Point,Building_CIR_Integration,Building_HeatMap,split_image,histogram,Line_Chart_Plot
from network.utils.tool import mse

dataset_path='./data'
dataset_input='input'
is_regeneration=False
output_times=2.5
output_dir=["./results/PIDRTN","./results/PIDRTN-A"]

#=====================

#image of input and labels

#Building grayscale image
Building_HeatMap(f"{dataset_path}/{dataset_input}",f"{dataset_path}/{dataset_input}_heatmap",regeneration=False,color_mode=1)


#Color images of CIR labels

#Building_HeatMap(f"{dataset_path}/output_32",f"{dataset_path}/output_32_heatmap",regeneration=False,times=output_times,regeneration_num=32)


#Color images of cumulative CIR labels

#Building_HeatMap(f"{dataset_path}/output_overlap_32",f"{dataset_path}/output_overlap_32_heatmap",regeneration=False,times=output_times,regeneration_num=32)


#CIR labels and Building

#Building_CIR_Integration(f"{dataset_path}/{dataset_input}_heatmap",f"{dataset_path}/output_32_heatmap",f"{dataset_path}/output_32_heatmap_integration",alpha=0.8,regeneration=False,regeneration_num=32)     


#cumulative CIR labels and Building

#Building_CIR_Integration(f"{dataset_path}/{dataset_input}_heatmap",f"{dataset_path}/output_overlap_32_heatmap",f"{dataset_path}/output_overlap_32_heatmap_integration",alpha=0.8,regeneration=False,regeneration_num=32)     

#=====================

#Color images of the results
for i in output_dir:
    Building_HeatMap(i,i+"_heatmap",regeneration=is_regeneration,regeneration_num=32)
    Building_CIR_Integration(f"{dataset_path}/{dataset_input}_heatmap",i+"_heatmap",i+"_integration",alpha=0.8,regeneration=is_regeneration,regeneration_num=32)     
    split_image(i,save_folder=i+"_split",regeneration=is_regeneration,regeneration_num=32)
    Building_HeatMap(i+"_split",i+"_split_heatmap",regeneration=is_regeneration,regeneration_num=32)
    Building_CIR_Integration(f"{dataset_path}/{dataset_input}_heatmap",i+"_split_heatmap",i+"_split_heatmap_integration",alpha=0.8,regeneration=is_regeneration,regeneration_num=32)   


#=====================

#CIR in Scenario 1

filename="data_162_-0.0608,51.4793,-0.0592,51.4802"
location=[280,170]

# filename="data_1842_-0.1085,51.5314,-0.1069,51.5322"
# location=[165,240]

label=['CIR from RT method','CIR from PIDRTN model','CIR from PIDRTN-A model']

path_plot_list=[dataset_path+f"/output_32/{filename}"]+output_dir[:]

for i in range(1,len(path_plot_list)):
    path_plot_list[i]=path_plot_list[i]+f"_split/{filename}"

CIR_Single_Point(path_plot_list,
                location=location,is_PartialEnlargedView=False,zone=[5,7],times=[1]+[output_times]*len(output_dir),label=label,color=['black','#e6846d','#339db5'])


# image = Image.open(dataset_path+f"/{dataset_input}/map_{filename[5:]}.png")
# image=np.array(image)
# image[location[0]-5:location[0]+5,location[1]-5:location[1]+5]=255
# image=Image.fromarray(image)
# image.show()

#=====================

#CIR in Scenario 2

# filename="data_162_-0.0608,51.4793,-0.0592,51.4802"
# location=[280,170]

filename="data_1842_-0.1085,51.5314,-0.1069,51.5322"
location=[165,240]

label=['CIR from RT method','CIR from PIDRTN model','CIR from PIDRTN-A model']

path_plot_list=[dataset_path+f"/output_32/{filename}"]+output_dir[:]

for i in range(1,len(path_plot_list)):
    path_plot_list[i]=path_plot_list[i]+f"_split/{filename}"

CIR_Single_Point(path_plot_list,
                location=location,is_PartialEnlargedView=False,zone=[5,7],times=[1]+[output_times]*len(output_dir),label=label,color=['black','#e6846d','#339db5'])


# image = Image.open(dataset_path+f"/{dataset_input}/map_{filename[5:]}.png")
# image=np.array(image)
# image[location[0]-5:location[0]+5,location[1]-5:location[1]+5]=255
# image=Image.fromarray(image)
# image.show()


#=====================
#MSE Histogram
output_dir_=["./results/PIDRTN-A","./results/PIDRTN","./results/U-net"]
error,error_max,error_min=mse(dataset_path+"/output_overlap_32",output_dir_,num=32,times=[output_times]*len(output_dir_))

# categories = ['8', '12', '16', '20', '24', '28', '32']
# categories = [9.375*8, 9.375*12, 9.375*16, 9.375*20, 9.375*24, 9.375*28, 9.375*32]
categories = ['75.0', '112.5', '150.0', '187.5', '225.0', '262.5', '300.0']

index=[7,11,15,19,23,27,31]
mean=[[] for i in range(len(output_dir_))]
data_max=[[] for i in range(len(output_dir_))]
data_min=[[] for i in range(len(output_dir_))]


for i in index:
    for j in range(len(output_dir_)):
        mean[j].append(error[j][i])
        data_max[j].append(error_max[j][i])
        data_min[j].append(error_min[j][i])

color=['#339db5','#e6846d','#F0DB8D']

label=['PIDRTN-A','PIDRTN','U-net']

histogram(categories,mean,data_max,data_min,color,label,y_interval=[0,0.05,0.10,0.15,0.20,0.25,0.30,0.4,0.5])

