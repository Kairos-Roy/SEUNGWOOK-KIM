from mrcnn import model as modellib
from mrcnn.config import Config
import os
import time

ROOT_DIR = "/home/ros2/tire_check180-1"  
DATASET_DIR = ROOT_DIR  
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "valid")
TEST_DIR = os.path.join(DATASET_DIR, "test")  
ANNOTATION_PATH_VAL = os.path.join(DATASET_DIR, "valid", "_annotations.coco.json")
ANNOTATION_PATH_TRAIN = os.path.join(DATASET_DIR, "train", "_annotations.coco.json")
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

class InferenceTireConfig(Config):
    NAME = "tire_check"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 2  # False, True
    BACKBONE = "resnet50" 
    IMAGE_MIN_DIM = 640 
    IMAGE_MAX_DIM = 640  
    IMAGE_RESIZE_MODE = "none"  
    DETECTION_MIN_CONFIDENCE = 0.6
    DETECTION_NMS_THRESHOLD = 0.2  
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)  
    MAX_GT_INSTANCES = 50
    DETECTION_MAX_INSTANCES = 200

config = InferenceTireConfig()
model = modellib.MaskRCNN(mode="inference", config=config, model_dir="/home/ros2/tire_check180-1")

weights_path = "/home/ros2/tire_check180-1/mask_rcnn_tire_weights_1206-4.h5"
model.load_weights(weights_path, by_name=True)

####Predict
import numpy as np
import cv2
import skimage
import matplotlib.pyplot as plt
from mrcnn.visualize import display_instances

class_names = ['BG', 'False', 'True']

shared_directory = "/home/ros2/ros2_ws/src/Mamot_pro"
output_directory = "/home/ros2/ros2_ws/src/Mamot_pro/MaskR"

while True:
    image_path = os.path.join(shared_directory, "Tire.jpg")
    if os.path.exists(image_path):  
        print("Tire image detected!")
        image = cv2.imread(image_path)
        
        if image is not None:
            print("Image loaded, starting analysis...")
            
            # Resized and Transfer skimage format 
            image_resized = skimage.transform.resize(image, (640, 640), mode='reflect', anti_aliasing=True)
            image_resized = (image_resized * 255).astype(np.uint8)

            # Result
            results = model.detect([image_resized], verbose=1)
            result = results[0]

            if len(result['class_ids']) == 0:
                print("No objects detected in the image!")
            else:
                print("Detected Class IDs:", result['class_ids'])
                print("Detected Scores:", result['scores'])
                
                first_class_id = result['class_ids'][0]
                output_path = os.path.join(output_directory, f"{first_class_id}.png")
                
                plt.figure(figsize=(10, 10))
                display_instances(
                    image_resized,  
                    result['rois'],  
                    result['masks'],  
                    result['class_ids'],  
                    class_names, 
                    scores=result['scores'],  
                    title="Mamot DeepLearning Report" 
                )
                plt.savefig(output_path)
                plt.close()
                               
                break

print("Mamot Deep Learning Finish")