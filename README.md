# detectron2-dfki
This repo is for evaluation purpose, build for instance segmentation task on EVICAN2 Dataset



##  Dependency Installation

pip install opencv-python
pip install pyyaml==5.3.1
pip install torch==1.8.1+cu101 torchvision==0.9.1+cu101 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.7/index.html
pip install matplotlib==3.2.2
pip install Pillow==9.2.0
pip install pycocotools==2.0.4

##  Dataset URL

Download the EVICAN2 dataset and place it in the project

https://edmond.mpdl.mpg.de/dataset.xhtml?persistentId=doi:10.17617/3.AJBV1S



##  Training Detectron2 Instance Segmentation Model Locally using .py

To train run: 
python main_train.py

To Evaluate run:
python main_eval.py


##  Training Detectron2 Instance Segmentation Model Locally using NoteBook

Run the Notebook below to run the train/eval locally 
Main_Detectron2.ipynb


##  Training Detectron2 Instance Segmentation Model Locally using Google Colab

Run the Notebook below to run the train/eval locally 
Colab/Main_Detectron2.ipynb



##  DOCKER

Due to time and hardware constraints, I could not complete the docker container building part.
(Working on it)






