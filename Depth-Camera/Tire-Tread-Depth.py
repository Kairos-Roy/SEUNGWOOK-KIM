import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import matplotlib.pyplot as plt
import threading
from sklearn.linear_model import RANSACRegressor

class Tire_depth_mamot(Node):
    def __init__(self):
        super().__init__('Tire_depth_mamot')
        self.depth_subscription = self.create_subscription(
            Image,
            '/camera/depth/image_raw',
            self.depth_image_callback,
            10)
        
        self.bridge = CvBridge()
        self.row_indices = [150]  
        self.depth_image = None
        self.file_counter = 1

    def depth_image_callback(self, msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, "32FC1")
        self.depth_received = True      
          
        normalized_depth_image = cv2.normalize(self.depth_image, None, 0, 255, cv2.NORM_MINMAX)
        depth_image_8bit = normalized_depth_image.astype(np.uint8)

        for x in range(0, depth_image_8bit.shape[1], 50): 
            cv2.line(depth_image_8bit, (x, 0), (x, depth_image_8bit.shape[0]), (255, 0, 0), 1)
            cv2.putText(depth_image_8bit, f'{x}', (x + 2, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("Tire Depth", depth_image_8bit)
        
        key = cv2.waitKey(1)
        if key == ord('q'): 
            self.get_logger().info("Finish")
            rclpy.shutdown()
        elif key == ord('c'):  
            self.capture_depth()
            

    def capture_depth(self):
        if self.depth_image is None:
            self.get_logger().info("No depth frame")
            return
        
        # 1. Capture Mulit images
        images_to_process = []
        for _ in range(3): 
            depth_image = self.depth_image
            images_to_process.append(depth_image)
        
        # 2. averaged_depth_image
        averaged_depth_image = np.mean(images_to_process, axis=0)
        filtered_depth_data = {row_index: [] for row_index in self.row_indices}
        
        # 3. Data Filtering & Save
        for row_index in self.row_indices:
            depth_row = averaged_depth_image[row_index, 190:450]  
            filtered_row = [
                (x+190, depth)  
                for x, depth in enumerate(depth_row)
                if 30.0 <= depth <= 250.0 
            ]
            filtered_depth_data[row_index].extend(filtered_row)

        file_name = f"Tire_depth_{self.file_counter}.txt"
        self.save_depth_data(filtered_depth_data, file_name)
     
        threading.Thread(target=self.plot_tread_data, args=(filtered_depth_data,)).start()
        self.file_counter += 1  

    def save_depth_data(self, data, filename):
        with open(filename, "w") as file:
            file.write("Y Row, Pixel X Coordinate, Depth Value:\n")
            for y, depth_data in data.items():
                for x, depth in depth_data:
                    file.write(f"{y},{x},{depth}\n")
        self.get_logger().info(f"Saved to {filename}")

    def plot_tread_data(self, filtered_depth_data):
        for row_index, depth_data in filtered_depth_data.items():
            x_coords = [x for x, _ in depth_data]
            z_values = [depth for _, depth in depth_data]

            # 1. Try RANSAC 
            z_corrected, _ = ransac_plane_fit(x_coords, z_values)  

            # 2. Seperate Surface & Groove + gradient 
            gradients = np.gradient(z_corrected)
            steep_indices = np.where(np.abs(gradients) > 7)[0] 
            
            # 3. Except behind
            valid_indices = [i for i in range(len(z_corrected)) if i not in steep_indices]
            x_valid = np.array(x_coords)[valid_indices]
            z_valid = np.array(z_corrected)[valid_indices]

            # 4. Seperate Data
            #threshold = np.percentile(z_valid, 50)  
            threshold = np.mean(z_valid)

            groove_indices = np.where((threshold + 5 > z_valid) & (z_valid> threshold + 1.5))[0]  
            surface_indices = np.where(z_valid <= threshold - 1)[0]  

            # 5. Calculate Depth 
            groove_depth = np.mean(z_valid[groove_indices]) 
            surface_depth = np.mean(z_valid[surface_indices]) 
            if groove_depth is not None and groove_depth is not None: 
                tread_depth = groove_depth - surface_depth  
                print(f"Tread_depth:",{tread_depth})
            else:
                tread_depth = 0
                print(f"Sorry try again")
                
            # 6. Visualize Data
            plt.figure(figsize=(12, 6))
            plt.axhline(y=threshold, color='green', linestyle='--', linewidth=1, label="Threshold") 
            plt.scatter(x_valid[groove_indices], z_valid[groove_indices], color="red", label="Groove")
            plt.scatter(x_valid[surface_indices], z_valid[surface_indices], color="blue", label="Surface")
            
            plt.plot(x_coords, z_corrected, label="Corrected Depth", color="black", linestyle='-', alpha=1.0)                        
            plt.title(f"Tire Tread Depth (y={row_index})\nTread Depth: {tread_depth:.2f}")
            plt.xlabel("X Coordinate")
            plt.ylabel("Depth Value")
            plt.legend()
            plt.grid()
            plt.savefig(f"Thread_Graph_{self.file_counter}.png")               
            plt.show()
         

def ransac_plane_fit(x_coords, z_values):
    # 1. numpy
    x_coords = np.array(x_coords).reshape(-1, 1)
    z_values = np.array(z_values)

    # 2. Filtering NaN & Inf data
    valid_indices = ~np.isnan(z_values) & ~np.isinf(z_values)
    x_coords = x_coords[valid_indices]
    z_values = z_values[valid_indices]

    # 3. Make RANSAC Model & Learning
    if len(x_coords) == 0 or len(z_values) == 0:
        raise ValueError("Filtered data is empty")
    
    ransac = RANSACRegressor()
    ransac.fit(x_coords, z_values)

    # 4. Calculate z_corrected 
    z_plane = ransac.predict(x_coords)
    z_corrected = z_values - z_plane 

    return z_corrected, z_plane


def main(args=None):
    rclpy.init(args=args)
    node = Tire_depth_mamot()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
