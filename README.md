# XPillars: Enhancing 3D Object Detection through Cross-Pillar Feature Fusion

**Lijuan Zhang, Zihan Fu, Zhiyi Li, Dongming Li**

## Introduction

This repo is the  implementation of paper: **XPillars: Enhancing 3D Object Detection through Cross-Pillar Feature Fusion** as well as based on the powerful open source point cloud detection framework [OpenPCDet](https://github.com/open-mmlab/OpenPCDet). Our XPillars achieves high performance on KITTI Dataset with real-time inference speed (65Hz). We have made every effort to ensure that the codebase is clean, concise, easily readable, and relies only on minimal dependencies.
<div align="center">
  <img src="docs/1.jpg" width="700"/>
</div>

- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Training & Testing](#training--testing)
- [Visualization](#visualization)
- [Citation](#citation)
- [Acknowledgement](#acknowledgement)

## Installation

### Requirements
All the codes are tested in the following environment:
* Ubuntu 22.04
* Python 3.10, 3.8
* PyTorch 2.4.1, 1.10
* CUDA 12.1, 11.3

The following environments are also available but never tested:
* Linux (tested on Ubuntu 14.04/16.04/18.04/20.04/21.04)
* Python 3.6+
* PyTorch 1.1 or higher (tested on PyTorch 1.1, 1,3, 1,5~1.10)
* CUDA 9.0 or higher (PyTorch 1.3+ needs CUDA 9.2+)

### Install 

a. Clone this repository.

b. Install the dependent libraries as follows:

* Install the SparseConv library, we use the implementation from [`[spconv]`](https://github.com/traveller59/spconv). 
    * If you use PyTorch 1.1, then make sure you install the `spconv v1.0` with ([commit 8da6f96](https://github.com/traveller59/spconv/tree/8da6f967fb9a054d8870c3515b1b44eca2103634)) instead of the latest one.
    * If you use PyTorch 1.3+, then you need to install the `spconv v1.2`. As mentioned by the author of [`spconv`](https://github.com/traveller59/spconv), you need to use their docker if you use PyTorch 1.4+. 
    * You could also install latest `spconv v2.x` with pip, see the official documents of [spconv](https://github.com/traveller59/spconv).
  
c. Install this library and its dependent libraries by running the following command:
```shell
python setup.py develop
```

## Data Preparation

* Please download the official [KITTI 3D object detection](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) dataset and organize the downloaded files as follows:

```
OpenPCDet
├── data
│   ├── kitti
│   │   │── ImageSets
│   │   │── training
│   │   │   ├──calib & velodyne & label_2 & image_2
│   │   │── testing
│   │   │   ├──calib & velodyne & image_2
├── pcdet
├── tools
```

* Generate the data infos by running the following command: 
```python 
python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml
```

## Training & Testing

```shell script
cd tools
python train.py
python test.py
```


## Visualization

### Requirements
It is recommended to create a new Conda environment to ensure compatibility with the required libraries. The following environment configuration has been tested and is known to work on Ubuntu 22.04:

```bash
conda create -n vis python=3.7
conda activate vis
pip install mayavi==4.7.4 scipy==1.7.3 pyqt5==5.15.6 vtk==8.1.2 opencv-python==4.1.2
```

### Directory Structure
The code expects the following directory structure under `XPillars/data/kitti/training`. 

You can get predict results by using `--save_to_file`:
```
training/
├── calib/
│   └── 000000.txt
├── image_2/
│   └── 000000.png
├── label_2/
│   └── 000000.txt
├── velodyne/
│   └── 000000.bin
└── pred/
    └── 000000.txt
```

### Usage
Run the script with desired arguments. Examples:
```bash
python kitti_vis/kitti_object.py -h # Get help
python kitti_vis/kitti_object.py --vis --show_image_with_boxes --show_lidar_with_boxes --ind 0  # Show image and LiDAR for the index 0
```

## Citation

If you find our work or this code useful in your research, please cite the manuscript associated with this repository. The paper, titled **"XPillars: Enhancing 3D Object Detection through Cross-Pillar Feature Fusion"**, has been submitted to _**The Visual Computer**_.

*Please note: Full citation details (volume, pages, DOI) will be updated here if and when the paper is accepted for publication.*

**BibTeX:**

```bibtex
@misc{XPillars,
  author       = {Lijuan Zhang, Zihan Fu, Zhiyi Li, and Dongming Li},
  title        = {Enhancing 3D Object Detection through Cross-Pillar Feature Fusion},
  year         = {2025},
  howpublished = {Manuscript submitted to The Visual Computer},
  note         = {Code available at: https://github.com/Isabelleux/XPillars.git}
}
```

## Acknowledgement
This repo is based on the open source project [OpenPCDet](https://github.com/open-mmlab/OpenPCDet).

This work was also supported by the National Natural Science Foundation of China (No. 62206257); "Light of the Taihu Lake" scientific and technological research project for Wuxi Science and Technology Development Fund (No. K20241044)；Wuxi University Research Start-up Fund for Introduced Talents (No.2023r004, 2023r006); Wuxi City Internet of Vehicles Key Laboratory.
