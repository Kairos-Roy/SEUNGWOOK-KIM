# Tire Tread Analysis with Mask R-CNN  

## Overview  
This repository demonstrates a method for analyzing tire tread conditions using Mask R-CNN. The project includes:  
1. **Surface Segmentation:** Labeling tire surfaces (excluding treads) and converting them into COCO format datasets.  
2. **Classification:** Fine-tuning Mask R-CNN to classify tires as **False (Severely Worn)** or **True (Good)** based on tread wear levels.  
3. **Inference:** Generating segmented masks and analyzing results for tire condition evaluation.  

## Workflow  
1. **Dataset Preparation:**  
   - Tire surfaces were labeled and converted to COCO format.  
2. **Model Training:**  
   - Pretrained Mask R-CNN was fine-tuned using tire wear annotations.  
3. **Validation:**  
   - Model was tested to accurately classify "Good" (True) vs. "Worn" (False) tires.  

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
