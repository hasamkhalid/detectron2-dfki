# Detectron2-dfki
This repo is for evaluation purpose, build for **instance-segmentation** task on **EVICAN2** Dataset



##  Dependency Installation

**Clone Detectron2**

git clone https://github.com/facebookresearch/detectron2.git

pip install opencv-python

pip install pyyaml==5.3.1

pip install torch==1.8.1+cu101 torchvision==0.9.1+cu101 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.7/index.html

pip install matplotlib==3.2.2

pip install Pillow==9.2.0

pip install pycocotools==2.0.4

##  Dataset URL

**Download the EVICAN2 dataset and place it in the project**

https://edmond.mpdl.mpg.de/dataset.xhtml?persistentId=doi:10.17617/3.AJBV1S



##  Training Detectron2 Instance Segmentation Model Locally using .py

**To train run:** 
python main_train.py

**To Evaluate run:**
python main_eval.py


##  Training Detectron2 Instance Segmentation Model Locally using NoteBook

**Run the Notebook below to run the train/eval locally**
Main_Detectron2.ipynb


##  Training Detectron2 Instance Segmentation Model Locally using Google Colab

**Run the Notebook below to run the train/eval locally**
Colab/Main_Detectron2.ipynb



##  DOCKER

**Clone Detectron2**

git clone https://github.com/facebookresearch/detectron2.git

**To Build Docker:**

**Build Docker Image:** docker compose build

**Run Docker Image:** docker run --gpus all -it detectron2

**To run already built Docker:**

**Download Docker image from this link:** https://bit.ly/3dYfOU5

**Load Docker Image:** docker load --input detectron2_docker.tar

**Run Docker Image:** docker run --gpus all -it detectron2






