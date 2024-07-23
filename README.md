# Physics-Informed Deep Ray Tracing Network

## Introduction
In modern communication systems, a profound grasp of the Channel Impulse Response (CIR) is pivotal for optimizing the design and functionality of algorithms and systems, especially the Multiple-Input-Multiple-Output (MIMO) technology. The conventional methods of gauging and modeling these channels rely on evaluating spatial points discretely sampled, resulting in limitations in acquiring pertinent channel information across a broader expanse of the area. 

To overcome these limitations, this paper embeds physical principles of electromagnetic wavefront propagation into data-driven deep learning models, achieving second-level regional CIR computing efficiency that is hundreds of times faster. 

The proposed Physics-Informed Deep Ray Tracing Network (PIDRTN) integrates multiple U-shaped Network (U-net) encoder-decoder blocks, capturing the radio wave propagation within a specific region surrounded with buildings, including two equivalent signal propagation directions in two-dimensional space and a signal intensity correction term. Then it employs a parameter-free nonlinear signal transmission module to emulate the physical laws of signal propagation as well as accurate CIRs from limited anchor locations, which will iteratively generates CIRs for various moments within the specified region subjected to enhancement and denoising operations. Meanwhile, the PIDRTN-A model, which utilizes anchor data to improve model accuracy, is proposed.

<br>
<div>
<img src="Figs/CIR_TO_Image.jpg" width="700px">
</div>
  <br>
<div>
<img src="Figs/Architecture.jpg" width="750px">
</div>
<br>

A dataset encompassing diverse fading scenarios is constructed by the Ray Tracing (RT) method. Extensive experimentation demonstrates that the proposed models adeptly learn direction and reflection properties.

## Requirements

Linux+pytorch(GPU)

- matplotlib==3.6.2
- numpy==1.24.3
- opencv-python==4.6.0.66
- Pillow==10.4.0
- scikit-learn==1.3.2
- torch==1.13.0
- torchsummary==1.5.1
- torchvision==0.14.0
- torchviz==0.0.2
- tqdm==4.64.1

## Datasets and Pretrained models
- There are only two examples here, please download more data from [here](https://drive.google.com/drive/folders/1rOjZoe6gM9DRt03JC5UouguWeE6HedLi?usp=drive_link).

Building information obtained from [OpenStreetMap](https://www.openstreetmap.org/). The labels are constructed by [Ray Tracing of Matlab](https://www.mathworks.com/help/comm/ref/rfprop.raytracing.html). 

Please unzip `data.zip` and put the data `input`, `output_32`,and `output_overlap_32` in `./data` folder.

Please put Pretrained models `PIDRTN.pth`, `PIDRTN-A.pth`,and `U-net.pth` in `./results/checkpoints_ddp` folder.

## Run

Please refer to  `./run.sh` for training and testing.

Please refer to  `./ResultsVisualization.py` for visualization.

## Results

### Scenario 1

<div>
<img src="Figs/CIR1.jpg" width="700px">
</div>


### Scenario 2

<div>
<img src="Figs/CIR2.jpg" width="700px">
</div>

## Citation
Please cite our paper when you use this code.
```
None
```

## Contact
Please contact wangshuchen191@mails.ucas.ac.cn if you have any question about this work.
