#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import numpy as np

class FingerBendPublisher(Node):
    def __init__(self):
        super().__init__('finger_bend_publisher')
        
        # Publisher for finger bend values
        self.finger_pub = self.create_publisher(
            Float32MultiArray,
            '/finger_bend_values',
            10
        )
        
        # Subscriber for camera image
        self.image_sub = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.image_callback,
            10
        )
        
        # Subscriber for depth image (if needed)
        self.depth_sub = self.create_subscription(
            Image,
            '/camera/camera/depth/image_rect_raw',
            self.depth_callback,
            10
        )
        
        # CV Bridge for image conversion
        self.bridge = CvBridge()
        
        # Store latest images
        self.color_image = None
        self.depth_image = None
        
        # Timer for publishing at regular intervals
        self.timer = self.create_timer(0.033, self.publish_finger_bend)  # ~30 Hz
        
        self.get_logger().info('Finger Bend Publisher Node started')
    
    def image_callback(self, msg):
        """Receive color image from RealSense"""
        try:
            self.color_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f'Error converting color image: {e}')
    
    def depth_callback(self, msg):
        """Receive depth image from RealSense"""
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, "passthrough")
        except Exception as e:
            self.get_logger().error(f'Error converting depth image: {e}')
    
    def calculate_finger_bend(self):
        """
        Calculate finger bend values from camera image.
        Replace this with your finger bend calculation code.
        
        Returns:
            list: Finger bend values [thumb, index, middle, ring, pinky]
                  Values should be between 0.0 (straight) and 1.0 (fully bent)
        """
        if self.color_image is None:
            return None
        
        # TODO: Replace this section with your finger bend calculation code
        # This is a placeholder that returns dummy values
        
        # Your code should process self.color_image and/or self.depth_image
        # and return the bend values for each finger
        
        # Example placeholder:
        finger_bends = [0.0, 0.0, 0.0, 0.0, 0.0]  # [thumb, index, middle, ring, pinky]
        
        # ADD YOUR FINGER DETECTION/CALCULATION CODE HERE
        # Example of what you might do:
        # 1. Detect hand in image
        # 2. Find finger keypoints
        # 3. Calculate angles between joints
        # 4. Normalize to 0-1 range
        
        return finger_bends
    
    def publish_finger_bend(self):
        """Publish finger bend values to ROS topic"""
        finger_bends = self.calculate_finger_bend()
        
        if finger_bends is not None:
            # Create message
            msg = Float32MultiArray()
            msg.data = finger_bends
            
            # Publish
            self.finger_pub.publish(msg)
            
            # Log values
            self.get_logger().info(
                f'Finger bends - Thumb: {finger_bends[0]:.2f}, '
                f'Index: {finger_bends[1]:.2f}, '
                f'Middle: {finger_bends[2]:.2f}, '
                f'Ring: {finger_bends[3]:.2f}, '
                f'Pinky: {finger_bends[4]:.2f}'
            )
        
        # Display image with visualization (optional)
        if self.color_image is not None:
            display_image = self.color_image.copy()
            
            # TODO: Add visualization of hand detection/finger tracking here
            # For example, draw skeleton, keypoints, or bend indicators
            
            cv2.imshow("Finger Tracking", display_image)
            cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = FingerBendPublisher()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()