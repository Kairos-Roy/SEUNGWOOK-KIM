#!/usr/bin/env python
# coding: utf-8

# In[1]:
import tensorflow as tf
print(tf.__version__)


# In[2]:
gpus = tf.config.experimental.list_physical_devices('GPU')
print("GPUs available:", gpus)


# In[3]:
get_ipython().system('pip install roboflow')
get_ipython().system('pip install pycocotools')
get_ipython().system('pip install git+https://github.com/akTwelve/Mask_RCNN.git')


# In[4]:
import pycocotools
print("pycocotools installed successfully")


# In[5]:
from roboflow import Roboflow
rf = Roboflow(api_key="")
project = rf.workspace("").project("tire_thread62")
version = project.version(2)
dataset = version.download("coco")


# In[6]:
import os

ROOT_DIR = "/home/ros2/tire_thread62-2"   
DATASET_DIR = ROOT_DIR  
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "valid")
TEST_DIR = os.path.join(DATASET_DIR, "test")  
ANNOTATION_PATH_VAL = os.path.join(DATASET_DIR, "valid", "_annotations.coco.json")
ANNOTATION_PATH_TRAIN = os.path.join(DATASET_DIR, "train", "_annotations.coco.json")


# In[7]:
# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


# In[8]:
import sys
sys.path.append(ROOT_DIR)

from mrcnn.config import Config
from mrcnn import model as modellib, utils
from pycocotools.coco import COCO


# In[9]:COCO API TEST
try:
    coco = COCO(ANNOTATION_PATH_TRAIN)
    print(f"COCO API loaded successfully with {len(coco.getCatIds())} categories.")
except Exception as e:
    print(f"Error loading COCO API: {e}")


# In[10]:Dir Check
print("Train directory exists:", os.path.exists(TRAIN_DIR))  
print("Validation directory exists:", os.path.exists(VAL_DIR))  
print("Test directory exists:", os.path.exists(TEST_DIR))  
print("Annotation file for train exists:", os.path.exists(ANNOTATION_PATH_TRAIN))  
print("Annotation file for validation exists:", os.path.exists(ANNOTATION_PATH_VAL))  


# In[11]:
class TireConfig(Config):
    NAME = "Tire_Mamot"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 2  # False, True
    STEPS_PER_EPOCH = 100
    VALIDATION_STEPS = 50
    BACKBONE = "resnet50"  # or "resnet101"
    IMAGE_MIN_DIM = 640
    IMAGE_MAX_DIM = 640
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)
    TRAIN_ROIS_PER_IMAGE = 50
    MAX_GT_INSTANCES = 50
    DETECTION_MAX_INSTANCES = 5
    DETECTION_MIN_CONFIDENCE = 0.7
    DETECTION_NMS_THRESHOLD = 0.4
    IMAGE_RESIZE_MODE = "none"
    LEARNING_RATE = 0.001


# In[12]:
class TireDataset(utils.Dataset):
    def load_tire(self, dataset_dir, subset):
        self.add_class("tire", 1, "False")
        self.add_class("tire", 2, "True")
        assert subset in ["train", "valid"]
        dataset_path = os.path.join(dataset_dir, subset)
        annotation_path = os.path.join(dataset_path, "_annotations.coco.json")

        coco = COCO(annotation_path)
        image_ids = list(coco.imgs.keys())

        for image_id in image_ids:
            self.add_image(
                "tire",
                image_id=image_id,
                path=os.path.join(dataset_path, coco.imgs[image_id]['file_name']),
                width=coco.imgs[image_id]["width"],
                height=coco.imgs[image_id]["height"],
                annotations=coco.loadAnns(coco.getAnnIds(imgIds=[image_id], iscrowd=False)),
            )

    def load_mask(self, image_id):
        annotation_path = os.path.join("/home/ros2/tire_check180-1/train/_annotations.coco.json")
        coco = COCO(annotation_path)
        image_info = self.image_info[image_id]
        annotations = image_info["annotations"]
        masks = []
        class_ids = []
        for annotation in annotations:
            category_id = annotation["category_id"]
            if category_id == 0:
                continue
            try:
                mask = coco.annToMask(annotation)
                print(f"Mask dtype: {mask.dtype}, shape: {mask.shape}")
                masks.append(mask)
                class_ids.append(category_id)
            except Exception as e:
                print(f"Annotation {annotation['id']} Error", e)
                
        if len(masks) == 0:
            print("No masks found!")
            return np.zeros((image_info["height"], image_info["width"], 0)), np.zeros((0,))
        else:
            masks = np.stack(masks, axis=-1)
            print(f"Stacked masks dtype: {masks.dtype}, shape: {masks.shape}")
            return masks.astype(np.uint8), np.array(class_ids, dtype=np.int32)
        
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        if info["source"] == "tire":
            return info["path"]
        else:
            return super(self.__class__, self).image_reference(image_id)


# In[13]:Ready DataSet
dataset_train = TireDataset()
dataset_train.load_tire(DATASET_DIR, "train")
dataset_train.prepare()
print('Train: %d' % len(dataset_train.image_ids))

dataset_val = TireDataset()
dataset_val.load_tire(DATASET_DIR, "valid")
dataset_val.prepare()
print('Test: %d' % len(dataset_val.image_ids))


# In[14]:Load and display random samples
from mrcnn import visualize
import numpy as np

image_ids = np.random.choice(dataset_train.image_ids, 4)
for image_id in image_ids:
    image = dataset_train.load_image(image_id)
    mask, class_ids = dataset_train.load_mask(image_id)
    
    image = image.astype(np.uint8)
    mask = mask.astype(np.uint8)  
    visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)


# In[15]:
import os
import json
dataset_dir = 'tire_check180-1'  
subset = 'train'  

dataset_path = os.path.join(dataset_dir, subset)
annotation_path = os.path.join(dataset_path, "_annotations.coco.json")

# Open Annotation json File
with open(annotation_path, 'r') as f:
    annotations = json.load(f)
    print(json.dumps(annotations, indent=4))

# In[16]:
from mrcnn.config import Config
config = TireConfig()
config.display()

# In[17]: Create Model
model = modellib.MaskRCNN(mode="training", config=config, model_dir=ROOT_DIR)
model.load_weights(COCO_MODEL_PATH, by_name=True, 
                   exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])

model.keras_model.summary()


# In[18]:
image = np.asarray(image, order='C', dtype=np.uint8)

# In[19]:train
model.train(dataset_train, dataset_val, epochs=20, layers='heads',learning_rate=0.001)

# In[22]:save weigths model
model.keras_model.save_weights('/home/ros2/tire_check180-1/mask_rcnn_tire_weights.h5')
