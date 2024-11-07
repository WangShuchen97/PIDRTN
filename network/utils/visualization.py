# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 17:55:12 2023

@author: Administrator
"""
import os
import copy
from tqdm import tqdm
from PIL import Image, ImageChops
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import  ConnectionPatch

def zone_and_linked(ax,axins,zone_left,zone_right,x,y,linked='bottom',
                    x_ratio=0.05,y_ratio=0.05):

    xlim_left = x[zone_left]-(x[zone_right]-x[zone_left])*x_ratio
    xlim_right = x[zone_right]+(x[zone_right]-x[zone_left])*x_ratio

    y_data = np.hstack([yi[zone_left:zone_right] for yi in y])
    ylim_bottom = np.min(y_data)-(np.max(y_data)-np.min(y_data))*y_ratio
    ylim_top = np.max(y_data)+(np.max(y_data)-np.min(y_data))*y_ratio
    if ylim_bottom==ylim_top:
        ylim_top+=0.001

    axins.set_xlim(xlim_left, xlim_right)
    axins.set_ylim(ylim_bottom, ylim_top)

    ax.plot([xlim_left,xlim_right,xlim_right,xlim_left,xlim_left],
            [ylim_bottom,ylim_bottom,ylim_top,ylim_top,ylim_bottom],"black")

    if linked == 'bottom':
        xyA_1, xyB_1 = (xlim_left,ylim_top), (xlim_left,ylim_bottom)
        xyA_2, xyB_2 = (xlim_right,ylim_top), (xlim_right,ylim_bottom)
    elif  linked == 'top':
        xyA_1, xyB_1 = (xlim_left,ylim_bottom), (xlim_left,ylim_top)
        xyA_2, xyB_2 = (xlim_right,ylim_bottom), (xlim_right,ylim_top)
    elif  linked == 'left':
        xyA_1, xyB_1 = (xlim_right,ylim_top), (xlim_left,ylim_top)
        xyA_2, xyB_2 = (xlim_right,ylim_bottom), (xlim_left,ylim_bottom)
    elif  linked == 'right':
        xyA_1, xyB_1 = (xlim_left,ylim_top), (xlim_right,ylim_top)
        xyA_2, xyB_2 = (xlim_left,ylim_bottom), (xlim_right,ylim_bottom)
        
    con = ConnectionPatch(xyA=xyA_1,xyB=xyB_1,coordsA="data",
                          coordsB="data",axesA=axins,axesB=ax)
    axins.add_artist(con)
    con = ConnectionPatch(xyA=xyA_2,xyB=xyB_2,coordsA="data",
                          coordsB="data",axesA=axins,axesB=ax)
    axins.add_artist(con)
    
def zone_and_Dashed(ax,zone_left,zone_right,x,y,x_ratio=0.05,y_ratio=0.05):

    xlim_left = x[zone_left]-(x[zone_right]-x[zone_left])*x_ratio
    xlim_right = x[zone_right]+(x[zone_right]-x[zone_left])*x_ratio

    y_data = np.hstack([yi[zone_left:zone_right] for yi in y])
    ylim_bottom = np.min(y_data)-(np.max(y_data)-np.min(y_data))*y_ratio
    ylim_top = np.max(y_data)+(np.max(y_data)-np.min(y_data))*y_ratio


    ax.plot([xlim_left,xlim_right,xlim_right,xlim_left,xlim_left],
            [ylim_bottom,ylim_bottom,ylim_top,ylim_top,ylim_bottom],'--r')

#CIR of Single point
def CIR_Single_Point(path_plot_list,location=[200,200],is_PartialEnlargedView=True,zone=None,label=[],times=[],color=[]):
    
    #location is the CIR location to be drawn
    
    #zone is the enlarged area of the partially enlarged image
    
    if len(label)==0:
        is_label=0
        label=['Y' for i in range(len(path_plot_list))]
    else:
        is_label=1
        
    if color==[]:
        color=['blue']*len(path_plot_list)
        
    if times==[]:
        times=[1 for i in range(len(path_plot_list))]
    fig, ax = plt.subplots(1,1,figsize=(15,8),dpi=1200)
    if is_PartialEnlargedView==True:
        #The position of a partially enlarged image in the original image
        axins = ax.inset_axes((0.4, 0.4, 0.4, 0.3))
        axins.tick_params(which='both',labelsize=15)
        axins.xaxis.get_offset_text().set_fontsize(15)
    for k,path_plot in enumerate(path_plot_list):
        num_time=len(os.listdir(path_plot))
        delay=[(i+1)*9.375e-9*1e9 for i in range(num_time)]
        temp=[]
        for i in range(num_time):
            img = Image.open(f"{path_plot}/{str(i).zfill(8)}.png") 
            img=np.array(img)
            temp.append(img/255)
        temp=np.array(temp)/times[k]
        if k==0:
            ax.plot(delay,temp[:,location[0],location[1]],label=label[k],color=color[k], linestyle='-', linewidth=2)
        else:
            ax.plot(delay,temp[:,location[0],location[1]],label=label[k],color=color[k], linestyle='--', linewidth=2)
        if is_PartialEnlargedView==True:
            axins.plot(delay,temp[:,location[0],location[1]],label=label[k])
    if is_PartialEnlargedView==True:
        zone_and_linked(ax, axins, zone[0], zone[1], delay , [temp[:,location[0],location[1]]], 'right')
    if is_label!=0:
        ax.legend(fontsize=20)
    ax.set_ylim([-0.002,0.04])
    ax.set_xlabel('Delay (ns)',fontdict={'size':20})
    ax.set_ylabel('Normalization CIR',fontdict={'size':20})
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout() 
    plt.tick_params(which='both',labelsize=20)
    plt.gca().xaxis.get_offset_text().set_fontsize(20)
    plt.show()

#Building and CIR integration
def Building_CIR_Integration(Building_folder,CIR_folder,save_folder,alpha=0.5,regeneration=False,regeneration_num=32):
    if not os.path.exists(f"{save_folder}"):
        os.mkdir(f"{save_folder}")
        
    folder_list=os.listdir(CIR_folder)
    with tqdm(folder_list, desc="Integration", total=len(folder_list)) as t:
        for i,_ in enumerate(t):
            t.set_postfix({"Filename":folder_list[i]})
            if not os.path.exists(f"{save_folder}/{folder_list[i]}"):
                os.mkdir(f"{save_folder}/{folder_list[i]}")
            else:
                if regeneration or len(os.listdir(f"{save_folder}/{folder_list[i]}"))!=regeneration_num:
                    pass
                else:
                    continue
            image1 = Image.open(f"{Building_folder}/map_{folder_list[i][5:]}.png")
            
            for j in range(len(os.listdir(f"{CIR_folder}/{folder_list[i]}"))):
                
                image2 = Image.open(f"{CIR_folder}/{folder_list[i]}/{str(j).zfill(8)}.png")

                image2 = image2.resize(image1.size)

                merged_image = Image.blend(image1, image2, alpha)

                merged_image.save(f"{save_folder}/{folder_list[i]}/{str(j).zfill(8)}.png")
            
            

def image_times(image,times):
        
    image_array = np.array(image)
    
    processed_image_array = np.clip(image_array.astype(np.uint16) * times, 0, 255).astype(np.uint8)
    
    processed_image = Image.fromarray(processed_image_array)
        
    return processed_image

def process_pixel(pixel, multiplier):
    
    return min(255, round(pixel * multiplier))


def Building_HeatMap(folder, save_folder="results/results_heatmap", regeneration=False,times=1,color_mode=0,regeneration_num=32):
    """
    Generate heatmaps from a folder of images.

    Args:
        folder (str): The path to the folder containing the images.
        save_folder (str, optional): The path to the folder where the heatmaps will be saved. Defaults to "results/HeatMap".
        regeneration (bool, optional): If True, regenerate the heatmaps even if they already exist. Defaults to False.
    """

    if not os.path.exists(f"{save_folder}"):
        os.mkdir(f"{save_folder}")
    

    #colors=["magenta","blueviolet","royalblue","aqua","springgreen","greenyellow","yellow","orangered","red","white"]

    #colors=["white","#FDCF6E","#45BD9B","#F05179","#2B557D"]
    
    if color_mode==0:
        #colors=["white","#B7D0EA","#407BD0","#203CD0","#000080","#000040"]
        colors=["white","#407BD0","#203CD0","#000080","#000040"]
    else:
        colors=["white","black"]
    cmap_custom=mcolors.LinearSegmentedColormap.from_list("mycmap",colors)

    folder_list = os.listdir(folder)
    save_folder_list=os.listdir(f"{save_folder}")
    
    with tqdm(folder_list, desc="HeatMap", total=len(folder_list)) as t:
        for i, _ in enumerate(t):
            t.set_postfix({"Filename": folder_list[i]})
            
            if os.path.isdir(f"{folder}/{folder_list[i]}"):
                if not os.path.exists(f"{save_folder}/{folder_list[i]}"):
                    os.mkdir(f"{save_folder}/{folder_list[i]}")
                else:
                    if regeneration or len(os.listdir(f"{save_folder}/{folder_list[i]}"))!=regeneration_num:
                        pass
                    else:
                        continue
            else:
                if regeneration or (folder_list[i] not in save_folder_list):
                    pass
                else:
                    continue
            if os.path.isdir(f"{folder}/{folder_list[i]}"):
                for j in range(len(os.listdir(f"{folder}/{folder_list[i]}"))):
                    image = Image.open(f"{folder}/{folder_list[i]}/{str(j).zfill(8)}.png")
 
                    
                    image = image_times(image,times)
                    
                    # Convert the grayscale image to a heatmap
                    plt.imshow(image, cmap=cmap_custom, interpolation='nearest',vmin=0, vmax=255,)
                    plt.axis('off')
                    plt.savefig(f"{save_folder}/{folder_list[i]}/{str(j).zfill(8)}.png", bbox_inches='tight', pad_inches=0)
                    plt.close()
            elif os.path.isfile(f"{folder}/{folder_list[i]}"):
                image = Image.open(f"{folder}/{folder_list[i]}")
                image = image_times(image,times)

                # Convert the grayscale image to a heatmap
                plt.imshow(image, cmap=cmap_custom, interpolation='nearest',vmin=0, vmax=255,)
                plt.axis('off')
                plt.savefig(f"{save_folder}/{folder_list[i]}", bbox_inches='tight', pad_inches=0)
                plt.close()


def split_image(folder, save_folder="./results/results_split", regeneration=False,regeneration_num=32):

    if not os.path.exists(f"{save_folder}"):
        os.mkdir(f"{save_folder}")

    folder_list = os.listdir(folder)
    with tqdm(folder_list, desc="split_image", total=len(folder_list)) as t:
        for i, _ in enumerate(t):
            t.set_postfix({"Filename": folder_list[i]})
            if not os.path.exists(f"{save_folder}/{folder_list[i]}"):
                os.mkdir(f"{save_folder}/{folder_list[i]}")
            else:
                if regeneration or len(os.listdir(f"{save_folder}/{folder_list[i]}"))!=regeneration_num:
                    pass
                else:
                    continue

            image_0 = Image.open(f"{folder}/{folder_list[i]}/{str(0).zfill(8)}.png")
            image_0.save(f"{save_folder}/{folder_list[i]}/{str(0).zfill(8)}.png")

            for j in range(1,len(os.listdir(f"{folder}/{folder_list[i]}"))):
                image_1 = Image.open(f"{folder}/{folder_list[i]}/{str(j).zfill(8)}.png")

                image = ImageChops.subtract(image_1, image_0)

                image.save(f"{save_folder}/{folder_list[i]}/{str(j).zfill(8)}.png")
                image_0=image_1
                
                

def map_y_to_ticks(y_value,y_ticks_values):
    for i in range(len(y_ticks_values) - 1):
        if y_value >= y_ticks_values[i] and y_value <= y_ticks_values[i+1]:
            return i + (y_value - y_ticks_values[i]) / (y_ticks_values[i+1] - y_ticks_values[i])
    return len(y_ticks_values)-1            
            
def histogram(categories,mean,data_max,data_min,color=['blue','orange'],label=['Mean Data 1','Mean Data 2'],y_interval=[]):
    if len(mean)==2:
        bar_width = 0.35 
        lin_width = 0.05 
    else:
        bar_width = 0.25 
        lin_width = 0.04 
    indices = np.arange(len(categories))  
    
    mean_=copy.deepcopy(mean)
    data_max_=copy.deepcopy(data_max)
    data_min_=copy.deepcopy(data_min)
    if y_interval!=[]:
        for i in range(len(mean)):
            for j in range(len(mean[0])):
                mean_[i][j]=map_y_to_ticks(mean[i][j],y_interval)
                data_max_[i][j]=map_y_to_ticks(data_max[i][j],y_interval)
                data_min_[i][j]=map_y_to_ticks(data_min[i][j],y_interval)
    n=len(mean)


    plt.figure(figsize=(15, 10),dpi=1200)

    for k in range(n):
        plt.bar(indices+k*bar_width, mean_[k], bar_width, color=color[k], alpha=0.8, label=label[k])

        #Connect the vertices of each category of columns with the same colored polylines
        plt.plot(indices+k*bar_width, mean_[k], linestyle='--', color=color[k],alpha=0.8, linewidth=2)  
        #Use short horizontal lines to mark the maximum and minimum values on each column, and connect them with vertical lines
        for i, (max_val, min_val) in enumerate(zip(data_max_[k], data_min_[k])):
            plt.plot([i - lin_width+k*bar_width, i + lin_width+k*bar_width], [max_val, max_val], color=color[k], linestyle='-', linewidth=2)  
            plt.plot([i - lin_width+k*bar_width, i + lin_width+k*bar_width], [min_val, min_val], color=color[k], linestyle='-', linewidth=2)  
            plt.plot([i+k*bar_width, i+k*bar_width], [min_val, max_val], color=color[k], linestyle='-', linewidth=2)  
            
        for i, value in enumerate(mean[k]):
            plt.text(i+k*bar_width-0.2, mean_[k][i]+0.1, f'{value:.4f}', ha='center',fontsize=15)


    plt.xticks(indices + bar_width / 2, categories)
    plt.xlabel('Delay (ns)',fontdict={'size':20})
    plt.ylabel('Normalized Root Mean Squared Error',fontdict={'size':20})

    plt.tick_params(which='both',labelsize=20)
    plt.gca().xaxis.get_offset_text().set_fontsize(20)
    plt.legend(prop={'size': 20},bbox_to_anchor=(0.25, 0.95))

    if y_interval!=[]:
        plt.yticks(range(len(y_interval)),y_interval)
        plt.ylim(0,len(y_interval)-1)
    else:
        plt.ylim(0,0.018)
    plt.grid(True)
    plt.tight_layout()
    plt.show() 
    


def Line_Chart_Plot(categories,y,color=[],label=['Data 1','Data 2'],y_interval=[]):

    plt.figure(figsize=(15, 10),dpi=1200)
    
    y_=copy.deepcopy(y)
    
    if y_interval!=[]:
        for i in range(len(y)):
            for j in range(len(y[0])):
                y_[i][j]=map_y_to_ticks(y_[i][j],y_interval)
                
    
    for k in range(len(y)):

        plt.plot(np.arange(len(categories)), y_[k], linestyle='-',marker='o',markersize=10,markerfacecolor='none', linewidth=2,label=label[k])  

    plt.xticks(np.arange(len(categories)),categories)
    plt.xlabel('Delay (ns)',fontdict={'size':20})
    plt.ylabel('Normalized Root Mean Squared Error',fontdict={'size':20})

    plt.tick_params(which='both',labelsize=20)
    plt.gca().xaxis.get_offset_text().set_fontsize(20)

    plt.legend(prop={'size': 20},bbox_to_anchor=(0.3, 0.95))
    
    if y_interval!=[]:
        plt.yticks(range(len(y_interval)),y_interval)
        plt.ylim(0,len(y_interval)-1)
    else:
        pass
        
    plt.grid(True)
    plt.tight_layout()
    plt.show() 
