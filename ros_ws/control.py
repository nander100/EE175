import pyrealsense2 as rs
import numpy as np
import cv2
import mediapipe as mp
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

'''
HAND LANDMARK INDEXES
Wrist: 0
Thumb: 1, 2, 3, 4
Index: 5, 6, 7, 8
Middle: 9, 10, 11, 12
Ring: 13, 14, 15, 16
Pinky: 17, 18, 19, 20
'''

class CameraControls(Node):
    def __init__(self):
        # Initialize ROS2 node
        super().__init__('hand_tracking_node')
        
        # Create publishers
        self.finger_bend_pub = self.create_publisher(Float32MultiArray, 'hand/finger_bend', 10)
        self.wrist_rotation_pub = self.create_publisher(Float32MultiArray, 'hand/wrist_rotation', 10)
        self.hand_position_pub = self.create_publisher(Float32MultiArray, 'hand/position', 10)
        
        print("Initializing camera...")
        # Initialize Hands
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Configure RealSense pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # Enable streams
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # Start streaming
        print("Starting RealSense camera...")
        self.pipeline.start(self.config)

        # Create align object to align depth to color
        align_to = rs.stream.color
        self.align = rs.align(align_to)

        # Initialize hand detection
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Create timer to publish at regular intervals (30 Hz)
        self.timer = self.create_timer(0.033, self.timer_callback)
        
        print("Camera initialized successfully!")
    
    def timer_callback(self):
        """Called periodically to read sensor data and publish"""
        try:
            bend = self.get_finger_bend()
            rotation = self.get_wrist_rotation()
            position = self.get_hand_position()
            
            if bend is not None:
                msg = Float32MultiArray()
                msg.data = [float(bend)]
                self.finger_bend_pub.publish(msg)
            
            if rotation is not None:
                msg = Float32MultiArray()
                msg.data = [float(rotation)]
                self.wrist_rotation_pub.publish(msg)
            
            if position is not None:
                msg = Float32MultiArray()
                msg.data = [float(position[0][0]), float(position[1][0]), float(position[2][0])]
                self.hand_position_pub.publish(msg)
                
            # Optional: log to console only when hand is detected
            if bend is not None and rotation is not None and position is not None:
                self.get_logger().info(
                    f"Bend: {bend:.1f}%, Rot: {rotation:.1f}°, Pos: [{position[0][0]:.3f}, {position[1][0]:.3f}, {position[2][0]:.3f}]"
                )
            
        except Exception as e:
            self.get_logger().error(f"Error in timer callback: {e}")
    
    def processHands(self):
        """Get the latest color frame from the camera and process for hand landmarks"""
        try:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            
            if not color_frame:
                return None
            
            color_image = np.asanyarray(color_frame.get_data())
            rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_image)
            
            if not results.multi_hand_landmarks:
                return None
            
            return results
        except Exception as e:
            self.get_logger().debug(f"Error processing hands: {e}")
            return None

    def get_finger_bend(self):
        """Calculate index finger bend percentage (0-100)"""
        try:
            results = self.processHands()
            
            if results is None:
                return None

            hand_landmarks = results.multi_hand_landmarks[0]
            
            mcp = hand_landmarks.landmark[5]
            pip = hand_landmarks.landmark[6]
            dip = hand_landmarks.landmark[7]
            
            v1 = np.array([pip.x - mcp.x, pip.y - mcp.y, pip.z - mcp.z])
            v2 = np.array([dip.x - pip.x, dip.y - pip.y, dip.z - pip.z])
            
            # Check for zero vectors
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            
            if norm1 == 0 or norm2 == 0:
                return None
            
            cos_angle = np.dot(v1, v2) / (norm1 * norm2)
            angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
            
            bend_percentage = ((180 - angle_deg) / 180) * 100
            
            return max(0, min(100, bend_percentage))
        except Exception as e:
            self.get_logger().debug(f"Error calculating finger bend: {e}")
            return None
    
    def get_wrist_rotation(self):
        """Calculate wrist rotation angle in degrees"""
        try:
            results = self.processHands()
            
            if results is None:
                return None
            
            hand_landmarks = results.multi_hand_landmarks[0]
            
            index_mcp = hand_landmarks.landmark[5]
            pinky_mcp = hand_landmarks.landmark[17]
            
            dx = pinky_mcp.x - index_mcp.x
            dy = pinky_mcp.y - index_mcp.y
            
            # Check for valid values
            if not (np.isfinite(dx) and np.isfinite(dy)):
                return None
            
            # Check if both dx and dy are essentially zero
            if abs(dx) < 1e-6 and abs(dy) < 1e-6:
                return None
            
            angle_rad = np.arctan2(dy, dx)
            angle_deg = np.degrees(angle_rad)
            
            # Normalize to -180 to 180 range
            angle_deg = (angle_deg % 360)
            if angle_deg > 180:
                angle_deg -= 360
            
            return angle_deg
        except Exception as e:
            self.get_logger().debug(f"Error calculating wrist rotation: {e}")
            return None
    
    def get_hand_position(self):
        """Find the 3D position of the hand wrist in meters"""
        try:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                return None
            
            color_image = np.asanyarray(color_frame.get_data())
            rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_image)
            
            if not results.multi_hand_landmarks:
                return None
            
            hand_landmarks = results.multi_hand_landmarks[0]
            wrist_landmark = hand_landmarks.landmark[0]
            
            img_h, img_w, _ = color_image.shape
            wrist_px = int(wrist_landmark.x * img_w)
            wrist_py = int(wrist_landmark.y * img_h)
            
            # Ensure pixel coordinates are within bounds
            if not (0 <= wrist_px < img_w and 0 <= wrist_py < img_h):
                return None
            
            depth_value = depth_frame.get_distance(wrist_px, wrist_py)
            
            # Check for valid depth
            if depth_value == 0 or not np.isfinite(depth_value):
                return None
            
            depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
            depth_point = rs.rs2_deproject_pixel_to_point(
                depth_intrin, [wrist_px, wrist_py], depth_value
            )
            
            # Check for valid 3D point
            if not all(np.isfinite(depth_point)):
                return None
            
            position_matrix = np.array([[depth_point[0]], 
                                        [depth_point[1]], 
                                        [depth_point[2]]])
            
            return position_matrix
        except Exception as e:
            self.get_logger().debug(f"Error calculating hand position: {e}")
            return None

    def __del__(self):
        """Stop the camera pipeline"""
        print("Stopping RealSense camera...")
        try:
            self.pipeline.stop()
        except:
            pass

def main(args=None):
    rclpy.init(args=args)
    
    try:
        camera_node = CameraControls()
        rclpy.spin(camera_node)
    except KeyboardInterrupt:
        pass
    finally:
        camera_node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()