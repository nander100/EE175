FROM osrf/ros:humble-desktop-full

# Install additional dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    ros-humble-gazebo-ros-pkgs \
    && rm -rf /var/lib/apt/lists/*

# Set up workspace
WORKDIR /ros2_ws
COPY . /ros2_ws/src

# Build workspace
RUN . /opt/ros/humble/setup.sh && \
    colcon build

# Source workspace on container start
RUN echo "source /ros2_ws/install/setup.bash" >> ~/.bashrc
