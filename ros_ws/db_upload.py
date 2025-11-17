import pyrealsense2 as rs
import numpy as np
import cv2
import mediapipe as mp
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from pymongo import MongoClient
import datetime
import os
import sys
import threading 
import queue 

'''
HAND LANDMARK INDEXES
Wrist: 0, Thumb: 1-4, Index: 5-8, Middle: 9-12, Ring: 13-16, Pinky: 17-20
'''

class CameraControls(Node):
    def __init__(self):
        super().__init__('hand_tracking_node')
        self.data_queue = queue.Queue(maxsize=100)
        self.db_thread_running = True
        
        # mongoDB connection
        self.mongo_client = None
        self.collection = None
        try:
            connection_string = os.environ.get('DB_URI')
            if not connection_string:
                self.get_logger().error("DB_URI environment variable not set. Shutting down.")
                raise ValueError("DB_URI not set")
            self.mongo_client = MongoClient(connection_string, serverSelectionTimeoutMS=5000)
            self.mongo_client.admin.command('ping')
            self.db = self.mongo_client['hand_tracking_db']
            self.collection = self.db['hand_movements']
            
            # Start the Database Writer Thread
            self.db_thread = threading.Thread(target=self.db_writer_worker, daemon=True)
            self.db_thread.start()
            self.get_logger().info("Successfully connected to MongoDB and started writer thread.")
        
        except Exception as e:
            self.get_logger().error(f"Could not connect to MongoDB: {e}")
            raise e

        # --- Publishers ---
        self.finger_bend_pub = self.create_publisher(Float32MultiArray, 'hand/finger_bend', 10)
        self.wrist_rotation_pub = self.create_publisher(Float32MultiArray, 'hand/wrist_rotation', 10)
        self.hand_position_pub = self.create_publisher(Float32MultiArray, 'hand/position', 10)
        
        # --- RealSense and MediaPipe Setup ---
        print("Initializing camera...")
        self.mp_hands = mp.solutions.hands
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        print("Starting RealSense camera...")
        self.pipeline.start(self.config)
        
        self.align = rs.align(rs.stream.color)
        self.hands = self.mp_hands.Hands(
            static_image_mode=False, max_num_hands=1,
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
        
        # --- ROS Timer ---
        self.timer = self.create_timer(0.033, self.timer_callback) # Aim for 30Hz
        print("Camera initialized successfully!")
    
    
    def db_writer_worker(self):
        """
        This function runs in a separate thread. 
        It waits for data to appear in the queue and uploads it to Mongo. 
        This loop is "slow" and doesn't block the main timer_callback. 
        """
        while self.db_thread_running:
            try:
                # Get data from the queue. Waits up to 1 sec if queue is empty
                data = self.data_queue.get(timeout=1.0)
                
                # If we get data, insert it.
                self.collection.insert_one(data)
                
                # Tell the queue we're done with this item
                self.data_queue.task_done()
                
            except queue.Empty:
                # This is normal, just means no data. Loop again.
                continue
            except Exception as e:
                self.get_logger().warn(f"DB writer thread error: {e}")
                
        print("DB writer thread shutting down.")

    
    def timer_callback(self):
        """
        This is the "fast loop". It does NO database work.
        It just gets data and puts it in the queue for the other thread.
        """
        try:
            frames = self.pipeline.wait_for_frames(1)
            aligned_frames = self.align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if not depth_frame or not color_frame: return

            color_image = np.asanyarray(color_frame.get_data())
            rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            
            # This is your main bottleneck!
            results = self.hands.process(rgb_image)
            
            if not results.multi_hand_landmarks:
                return
            
            hand_landmarks = results.multi_hand_landmarks[0]

            # --- Calculate all values ---
            bend = self.get_finger_bend_calc(hand_landmarks)
            rotation = self.get_wrist_rotation_calc(hand_landmarks)
            position = self.get_hand_position_calc(hand_landmarks, depth_frame, color_image.shape)
        
            if bend is not None and rotation is not None and position is not None:
                self.get_logger().info(
                    f"Bend: {bend:.1f}%, Rot: {rotation:.1f}°, Pos: [{position[0][0]:.3f}, {position[1][0]:.3f}, {position[2][0]:.3f}]"
                )
                
                hand_data_doc = {
                    "timestamp": datetime.datetime.utcnow(),
                    "bend_percentage": bend,
                    "wrist_rotation_deg": rotation,
                    "hand_position_m": {
                        "x": float(position[0][0]),
                        "y": float(position[1][0]),
                        "z": float(position[2][0])
                    }
                }
                
                try:
                    self.data_queue.put_nowait(hand_data_doc)
                except queue.Full:
                    self.get_logger().warn("Data queue is full. Dropping frame.")

        except RuntimeError as e:
            self.get_logger().error(f"RealSense runtime error: {e}")
        except Exception as e:
            self.get_logger().error(f"Error in timer callback: {e}")
    
    def get_finger_bend_calc(self, hand_landmarks):
        try:
            mcp = hand_landmarks.landmark[5]
            pip = hand_landmarks.landmark[6]
            dip = hand_landmarks.landmark[7]
            v1 = np.array([pip.x - mcp.x, pip.y - mcp.y, pip.z - mcp.z])
            v2 = np.array([dip.x - pip.x, dip.y - pip.y, dip.z - pip.z])
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            if norm1 == 0 or norm2 == 0: return None
            cos_angle = np.dot(v1, v2) / (norm1 * norm2)
            angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
            bend_percentage = ((180 - angle_deg) / 180) * 100
            return max(0, min(100, bend_percentage))
        except Exception as e:
            self.get_logger().debug(f"Error calculating finger bend: {e}")
            return None
    
    def get_wrist_rotation_calc(self, hand_landmarks):
        try:
            index_mcp = hand_landmarks.landmark[5]
            pinky_mcp = hand_landmarks.landmark[17]
            dx = pinky_mcp.x - index_mcp.x
            dy = pinky_mcp.y - index_mcp.y
            if not (np.isfinite(dx) and np.isfinite(dy)): return None
            if abs(dx) < 1e-6 and abs(dy) < 1e-6: return None
            angle_rad = np.arctan2(dy, dx)
            angle_deg = np.degrees(angle_rad)
            angle_deg = (angle_deg % 360)
            if angle_deg > 180: angle_deg -= 360
            return angle_deg
        except Exception as e:
            self.get_logger().debug(f"Error calculating wrist rotation: {e}")
            return None
    
    def get_hand_position_calc(self, hand_landmarks, depth_frame, image_shape):
        try:
            wrist_landmark = hand_landmarks.landmark[0]
            img_h, img_w, _ = image_shape
            wrist_px = int(wrist_landmark.x * img_w)
            wrist_py = int(wrist_landmark.y * img_h)
            if not (0 <= wrist_px < img_w and 0 <= wrist_py < img_h): return None
            depth_value = depth_frame.get_distance(wrist_px, wrist_py)
            if depth_value == 0 or not np.isfinite(depth_value): return None
            depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
            depth_point = rs.rs2_deproject_pixel_to_point(
                depth_intrin, [wrist_px, wrist_py], depth_value
            )
            if not all(np.isfinite(depth_point)): return None
            position_matrix = np.array([[depth_point[0]], [depth_point[1]], [depth_point[2]]])
            return position_matrix
        except Exception as e:
            self.get_logger().debug(f"Error calculating hand position: {e}")
            return None

    def stop_node(self):
        """Custom shutdown function."""
        print("Stopping RealSense camera...")
        try:
            self.pipeline.stop()
        except:
            pass
        
        print("Stopping DB writer thread...")
        self.db_thread_running = False
        if hasattr(self, 'db_thread'):
            self.db_thread.join(timeout=2.0)
        
        if self.mongo_client:
            print("Closing MongoDB connection...")
            self.mongo_client.close()
            
        self.destroy_node()


def main(args=None):
    rclpy.init(args=args)
    camera_node = None
    try:
        camera_node = CameraControls()
        rclpy.spin(camera_node)
    except KeyboardInterrupt:
        print("\nKeyboard interrupt detected, shutting down...")
    except Exception as e:
        print(f"Failed to initialize node: {e}")
    finally:
        if camera_node:
            camera_node.stop_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()