#!/bin/bash

#=================TEST=====================
python ./main.py --mode "test" --model_name "PIDRTN"

python ./main.py --mode "test" --model_name "PIDRTN-A"

python ./main.py --mode "test" --model_name "U-net"


#==========================================


#=============Visualization================

#For more visualizations, please refer to ./ResultsVisualization.py

#python ./ResultsVisualization.py

#==========================================




#=============Train Examples================

#python ./main.py --mode "train" --model_name "PIDRTN" --epochs 1 --save_name "train_PIDRTN"

#python ./main.py --mode "train" --model_name "PIDRTN-A" --epochs 1 --save_name "train_PIDRTN-A"

#python ./main.py --mode "train" --model_name "U-net" --epochs 1 --save_name "train_U-net"

#==========================================


