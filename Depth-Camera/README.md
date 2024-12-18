# Tire-Tread-Analysis-with-Depth-Camera

## Overview
This project is designed to capture depth data from a camera feed, process it to measure the depth of tire treads, and save the processed data for further analysis. The system uses an Astra U3 depth camera for capturing depth images and a ROS2 node to handle the image processing, depth calculation, and visualization of the results. The tread depth is calculated by analyzing the depth data from the tire and using a RANSAC model to fit a plane to the data points. The system then separates the surface and groove of the tire to calculate the tread depth.

## Features
- **Depth Image Capture**: The system captures depth data from the Astra U3 depth camera.
- **RANSAC Plane Fitting**: The RANSAC algorithm is used to fit a plane to the captured data and correct the depth measurements for tire tread analysis.
- **Tread Depth Calculation**: It separates the groove and surface of the tire to calculate the tread depth.
- **Data Filtering & Visualization**: Data points are filtered based on their depth values, and the results are visualized using matplotlib. The system identifies the surface and groove regions of the tire.
- **ROS2 Integration**: ROS2 is used for communication between the camera feed and processing components. This ensures that the system is modular and extensible.

## Requirements
- **ROS2 Humble**
- **Ubuntu 22.04**
- **Astra U3 Depth Camera**
- **Python Libraries**:
    - `rclpy`: ROS2 Python client library
    - `sensor_msgs`: ROS2 message types for sensor data
    - `cv_bridge`: Convert ROS images to OpenCV format
    - `opencv-python`: For image processing
    - `numpy`: Numerical operations
    - `matplotlib`: For visualizing depth data
    - `sklearn`: For RANSAC model fitting
