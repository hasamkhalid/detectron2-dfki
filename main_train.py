
import cv2
from tqdm import tqdm
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

from pycocotools.coco import COCO
from pycocotools import mask as mutils

from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetMapper, build_detection_test_loader
from detectron2.engine import DefaultTrainer
from LossEvalHook import LossEvalHook 
import os
import json
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

import torch
from detectron2.utils.visualizer import ColorMode
from detectron2.utils.visualizer import Visualizer
import cv2
from google.colab.patches import cv2_imshow

print(torch.__version__, torch.cuda.is_available())

root_dir = 'dataverse_files/'
annotations_dir = 'annotations_EVICAN2/'

#@title Calculate bbox and save new coco json

json_files = os.listdir(root_dir+annotations_dir)
for path in json_files:
  path = root_dir+annotations_dir+path
  f = open(path)
  anns = json.load(f)
  coco=COCO(path)
  # convert masks to annotations
  for i in range(len(anns['annotations'])):
    bbox = list(map(int,mutils.toBbox(coco.annToRLE(anns['annotations'][i])).tolist()))
    anns['annotations'][i]['bbox'] = bbox
    anns['annotations'][i]['area'] = bbox[2]*bbox[3]

  with open(path.split('.')[0]+'_bbox.json', 'w') as f:
    json.dump(anns, f)





register_coco_instances("my_dataset_train", {}, root_dir+annotations_dir+"instances_train2019_EVICAN2_bbox.json", root_dir+'Images/EVICAN_train2019')
register_coco_instances("my_dataset_val", {}, root_dir+annotations_dir+"instances_val2019_EVICAN2_bbox.json", root_dir+'Images/EVICAN_val2019')
register_coco_instances("my_dataset_eval", {}, root_dir+annotations_dir+"instances_eval2019_medium_EVICAN2_bbox.json", root_dir+'Images/EVICAN_eval2019')



class CustomTrainer(DefaultTrainer):
    """
    Custom Trainer deriving from the "DefaultTrainer"
    Overloads build_hooks to add a hook to calculate validation loss on the test set during training.
    """

    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1, LossEvalHook(
            100, # Frequency of calculation - every 100 iterations here
            self.model,
            build_detection_test_loader(
                self.cfg,
                self.cfg.DATASETS.TEST[0],
                DatasetMapper(self.cfg, True)
            )
        ))

        return hooks


cfg = get_cfg()
cfg.OUTPUT_DIR = "output/"
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

cfg.merge_from_file("detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")

cfg.DATASETS.TRAIN = ("my_dataset_train",)
cfg.DATASETS.TEST = ("my_dataset_val",)
cfg.TEST.EVAL_PERIOD = 100
cfg.DATALOADER.NUM_WORKERS = 2

cfg.MODEL.WEIGHTS = ("detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")

cfg.INPUT.MAX_SIZE_TEST = 20000
cfg.INPUT.MAX_SIZE_TRAIN = 20000
cfg.SOLVER.IMS_PER_BATCH = 2   # increase this for more speed, but it will need more vram
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
#cfg.SOLVER.BASE_LR = 0.000001  # pick a good LR /content/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml
cfg.SOLVER.MAX_ITER = 3000    # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 4096   # or 128 for example
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2 
cfg.SNAPSHOT_ITERS = 50 # when checkpoint gets created
cfg.SOLVER.CHECKPOINT_PERIOD = 50

# print(cfg.dump())  
with open("output/custom_mask_rcnn_X_101_32x8d_FPN_3x.yaml", "w") as f:
    f.write(cfg.dump())
    f.close()

trainer = CustomTrainer(cfg) 
# trainer.resume_or_load(resume=False)
trainer.train()


