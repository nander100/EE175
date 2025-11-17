FROM osrf/ros:humble-desktop-full

# Install additional dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    ros-humble-gazebo-ros-pkgs \
    && rm -rf /var/lib/apt/lists/*

# Set up workspace
WORKDIR /ros_ws
COPY . /ros_ws/src

# Build workspace
RUN . /opt/ros/humble/setup.sh && \
    colcon build

RUN pip install pyrealsense2 mediapipe pymongo

# Source workspace on container start
RUN echo "source /ros_ws/install/setup.bash" >> ~/.bashrc