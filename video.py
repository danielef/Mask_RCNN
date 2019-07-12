import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import cv2
import argparse

# Root directory of the project
ROOT_DIR = os.path.abspath(".")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco

# %matplotlib inline 

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

# Load Video

parser = argparse.ArgumentParser()
parser.add_argument('-v','--video',required=True,help='video file to detect')
args = parser.parse_args()

video = args.video # '/home/quantum/Videos/20190710_133650_119D_ACCC8EB7D10E.mkv'
print('Reading video: {}'.format(video))

cap = cv2.VideoCapture(video)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
print('size: {}x{}, fps: {}, lenght: {}'.format(width, height, fps, length))

# Video Output
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('{}-coco.avi'.format(video[:-4]), fourcc, fps, (int(width), int(height)))
ready = 0
colors = visualize.random_colors(80)
colors[1] = (255/255, 0/255, 0/255)
colors[3] = (255/255, 69/255, 0/255)
colors[6] = (0/255, 255/255, 0/255)
colors[10] = (0/255, 0/255, 255/255)

while(cap.isOpened()):
  # Capture frame-by-frame
  ret, image = cap.read()
  if ret == True:
 
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Run detection
    results = model.detect([image], verbose=1)

    r = results[0]
    fig, ax = visualize.display_instances(image, 
                                         r['rois'], 
                                         r['masks'], 
                                         r['class_ids'], 
                                         class_names, 
                                         r['scores'],
                                         colors=colors, plt_show=False, figsize=(34.5, 34.5))
    
    fig.savefig('/tmp/render.jpg', format='jpeg', dpi=fig.dpi, pad_inches=0, quality=100)
    data = cv2.imread('/tmp/render.jpg')
    y = 694# 432
    h = 1080
    x = 311
    w = 1920
    crop_img = data[y:y+h, x:x+w]
    out.write(crop_img)
    
    if ready < 50:
        ready += 1
    else:
        break
  # Break the loop
  else: 
    break

out.release()
cap.release()
