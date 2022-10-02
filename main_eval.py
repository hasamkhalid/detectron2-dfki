
from pycocotools.coco import COCO
from detectron2.data.datasets import register_coco_instances
import os
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
import torch
from detectron2.utils.visualizer import ColorMode
from detectron2.utils.visualizer import Visualizer
from google.colab.patches import cv2_imshow
from detectron2.utils.visualizer import ColorMode
from detectron2.utils.visualizer import Visualizer
import cv2
from detectron2.evaluation import COCOEvaluator, inference_on_dataset


print(torch.__version__, torch.cuda.is_available())

root_dir = 'dataverse_files/'
annotations_dir = 'annotations_EVICAN2/'

register_coco_instances("my_dataset_eval", {}, root_dir+annotations_dir+"instances_eval2019_medium_EVICAN2_bbox.json", root_dir+'Images/EVICAN_eval2019')


cfg.MODEL.WEIGHTS = 'output/model_best.pth'
#@title Loading model for Evaluation on Eval/Test Dataset 

cfg = get_cfg()
cfg.OUTPUT_DIR = "output/"
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)


cfg.merge_from_file("detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = ("detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
cfg.INPUT.MAX_SIZE_TEST = 20000
cfg.SOLVER.IMS_PER_BATCH = 2   # increase this for more speed, but it will need more vram
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 4096   # or 128 for example
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2 

cfg.MODEL.WEIGHTS = 'output/model_best.pth'

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.00  # set the testing threshold for this model
cfg.DATASETS.TEST = ("my_dataset_eval", )
predictor = DefaultPredictor(cfg)


#@title do detection and display result

os.makedirs('out_images/', exist_ok=True)
evalImages = 'dataverse_files/Images/EVICAN_eval2019/'
for imName in os.listdir(evalImages):
    im = cv2.imread(evalImages+imName)
    outputs = predictor(im)
    # print(outputs)
    v = Visualizer(im[:, :, ::-1],
                    metadata=None, 
                    scale=30, 
                    instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    out_file_name = 'out_images/'+imName
    cv2.imwrite(out_file_name, cv2.resize(v.get_image()[:, :, ::-1], (960, 540)))




evaluator = COCOEvaluator("my_dataset_eval", cfg, False, output_dir="output/")
val_loader = build_detection_test_loader(cfg, "my_dataset_eval")
inference_on_dataset(predictor.model, val_loader, evaluator)



# Loading and preparing results...
# DONE (t=0.00s)
# creating index...
# index created!
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.002
#  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.011
#  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.000
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.009
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.005
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.001
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.001
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.012
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.027
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.066
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.030
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.037
# Loading and preparing results...
# DONE (t=0.03s)
# creating index...
# index created!
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.000
#  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.001
#  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.000
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.005
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.000
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.000
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.000
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.004
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.008
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.038
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.006
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.010
# OrderedDict([('bbox',
#               {'AP': 0.17254292975408037,
#                'AP50': 1.1195842724364529,
#                'AP75': 0.0,
#                'APs': 0.8763554926921263,
#                'APm': 0.5047392466930145,
#                'APl': 0.07444822860965505,
#                'AP-Cell': 0.33549550320508437,
#                'AP-Nucleus': 0.009590356303076326}),
#              ('segm',
#               {'AP': 0.014488106223065712,
#                'AP50': 0.09535973043070121,
#                'AP75': 0.0,
#                'APs': 0.4555634134842054,
#                'APm': 0.007241045682282121,
#                'APl': 0.012912911569183397,
#                'AP-Cell': 0.027541556783242587,
#                'AP-Nucleus': 0.0014346556628888325})])