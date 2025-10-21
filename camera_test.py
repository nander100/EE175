import pyrealsense2 as rs
import numpy as np
import cv2
import mediapipe as mp



# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Configure RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()

# Enable streams
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
print("Starting RealSense camera...")
pipeline.start(config)

# Create align object to align depth to color
align_to = rs.stream.color
align = rs.align(align_to)

# Initialize hand detection
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

print("Hand detection initialized. Press 'q' to quit.")

try:
    while True:
        # This line of code 
        frames = pipeline.wait_for_frames()
        
        # Align depth frame to color frame
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        

        if not depth_frame or not color_frame:
            continue
        
        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        # Convert BGR to RGB for MediaPipe
        rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        
        # Process hand detection
        results = hands.process(rgb_image)
        
        # Create a copy for drawing
        annotated_image = color_image.copy()
        
        # Draw hand landmarks and connections
        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Draw landmarks
                mp_drawing.draw_landmarks(
                    annotated_image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Get hand label (Left/Right)
                hand_label = results.multi_handedness[hand_idx].classification[0].label
                
                # Draw bounding box around hand
                h, w, _ = annotated_image.shape
                x_coords = [lm.x for lm in hand_landmarks.landmark]
                y_coords = [lm.y for lm in hand_landmarks.landmark]
                
                x_min = int(min(x_coords) * w)
                x_max = int(max(x_coords) * w)
                y_min = int(min(y_coords) * h)
                y_max = int(max(y_coords) * h)
                
                # Draw bounding box
                cv2.rectangle(annotated_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                
                # Calculate hand center (average of all landmarks)
                center_x = sum([lm.x for lm in hand_landmarks.landmark]) / len(hand_landmarks.landmark)
                center_y = sum([lm.y for lm in hand_landmarks.landmark]) / len(hand_landmarks.landmark)
                
                # Convert to pixel coordinates
                center_px = int(center_x * w)
                center_py = int(center_y * h)
                
                # Get depth value at center (in meters)
                depth_value = 0.0
                if 0 <= center_px < w and 0 <= center_py < h:
                    depth_value = depth_frame.get_distance(center_px, center_py)
                    
                    # Get 3D coordinates using RealSense intrinsics
                    depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
                    depth_point = rs.rs2_deproject_pixel_to_point(
                        depth_intrin, [center_px, center_py], depth_value
                    )
                    
                    # Create 3x1 position matrix [X, Y, Z] in meters
                    position_matrix = np.array([[depth_point[0]], 
                                               [depth_point[1]], 
                                               [depth_point[2]]])
                    
                    # Display hand info and position
                    info_text = f"{hand_label} Hand"
                    cv2.putText(annotated_image, info_text, 
                               (x_min, y_min - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Draw center point
                    cv2.circle(annotated_image, (center_px, center_py), 8, (0, 0, 255), -1)
                    cv2.circle(annotated_image, (center_px, center_py), 10, (255, 255, 255), 2)
                    
                    # Print 3D position to console
                    print(f"\n{hand_label} Hand Position (3x1 matrix):")
                    print(position_matrix)
                    print(f"X: {position_matrix[0][0]:.4f}m")
                    print(f"Y: {position_matrix[1][0]:.4f}m") 
                    print(f"Z: {position_matrix[2][0]:.4f}m")
                    
                    # Calculate index finger bend angle
                    # Index finger landmarks: 5 (MCP), 6 (PIP), 7 (DIP), 8 (TIP)
                    mcp = hand_landmarks.landmark[5]  # Metacarpophalangeal joint
                    pip = hand_landmarks.landmark[6]  # Proximal interphalangeal joint
                    dip = hand_landmarks.landmark[7]  # Distal interphalangeal joint
                    tip = hand_landmarks.landmark[8]  # Finger tip
                    
                    # Calculate vectors
                    v1 = np.array([pip.x - mcp.x, pip.y - mcp.y, pip.z - mcp.z])
                    v2 = np.array([dip.x - pip.x, dip.y - pip.y, dip.z - pip.z])
                    
                    # Calculate angle between vectors using dot product
                    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Clamp to valid range
                    angle_rad = np.arccos(cos_angle)
                    angle_deg = np.degrees(angle_rad)
                    
                    # Calculate bend percentage (180° = straight, 0° = fully bent)
                    bend_percentage = ((180 - angle_deg) / 180) * 100
                    
                    print(f"Index Finger Bend: {angle_deg:.1f}° ({bend_percentage:.1f}% bent)")
                    
                    # Display bend info on image
                    bend_text = f"Index: {angle_deg:.0f}deg ({bend_percentage:.0f}%)"
                    cv2.putText(annotated_image, bend_text,
                               (x_min, y_max + 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 2)
                
                # Create hand contour outline
                points = []
                for lm in hand_landmarks.landmark:
                    px = int(lm.x * w)
                    py = int(lm.y * h)
                    points.append([px, py])
                
                # Create convex hull for hand outline
                points = np.array(points)
                hull = cv2.convexHull(points)
                cv2.polylines(annotated_image, [hull], True, (255, 0, 0), 2)
        
        # Create depth colormap
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03),
            cv2.COLORMAP_JET
        )
        
        # Stack images side by side
        images = np.hstack((annotated_image, depth_colormap))
        
        # Add instructions
        cv2.putText(images, "Press 'q' to quit", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display
        cv2.imshow('RealSense D455 - Hand Detection & Depth', images)
        
        # Break loop on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Cleanup
    pipeline.stop()
    hands.close()
    cv2.destroyAllWindows()
    print("Cleaned up resources.")
