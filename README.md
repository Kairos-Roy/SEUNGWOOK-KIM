# Tire Tread Analysis with Mask R-CNN  

## Overview  
This repository demonstrates a method for analyzing tire tread conditions using Mask R-CNN. The project includes:  
1. **Surface Segmentation:** Labeling tire surfaces (excluding treads) and converting them into COCO format datasets.  
2. **Classification:** Fine-tuning Mask R-CNN to classify tires as **False (Severely Worn)** or **True (Good)** based on tread wear levels.  
3. **Inference:** Using the trained model to generate segmented masks and analyze tire conditions.  

## Workflow  
1. **Dataset Preparation:**  
   - Tire surfaces were labeled and converted to COCO format.  
2. **Model Training:**  
   - Pretrained Mask R-CNN initialized with COCO weights was fine-tuned using tire wear annotations.  
   - **Backbone Architecture:** ResNet50 was used as the feature extractor for the model.  
   - The final weights file (`best_model_weights.h5`) was saved at the end of the training process.  
3. **Inference:**  
   - The saved weights file is loaded for generating segmentation masks and classifying tire conditions as "Good" (True) or "Worn" (False).  

## Environment Setup  
This project was developed and tested in the following environment:  
- **Operating System:** Ubuntu 22.04  
- **Development Tools:** Jupyter Notebook  
- **Dependencies:**  
  ```bash  
  pip install tensorflow==1.14.0  
  pip install numpy==1.21.6  
  pip install keras==2.2.4  
  pip install keras-applications==1.0.8  
  pip install scikit-image==0.16.2  
