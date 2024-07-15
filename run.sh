#!/bin/bash

#=================TEST=====================
python ./main.py --mode "test" --model_name "deepraytracing"

python ./main.py --mode "test" --model_name "deepraytracing_A"

python ./main.py --mode "test" --model_name "unet"


#==========================================


#=============Visualization================

#For more visualizations, please refer to ./ResultsVisualization.py

#python ./ResultsVisualization.py

#==========================================




#=============Train Examples================

#python ./main.py --mode "train" --model_name "deepraytracing" --epochs 1 --save_name "train_DRTN"

#python ./main.py --mode "train" --model_name "deepraytracing_A" --epochs 1 --save_name "train_DRTN-A"

#python ./main.py --mode "train" --model_name "unet" --epochs 1 --save_name "train_Unet"

#==========================================


