# Dockerfile
FROM ubuntu:20.04

# Prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Update and install essential dependencies
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3-pip \
    python3.8-venv \
    wget \
    git \
    build-essential \
    cmake \
    libboost-all-dev \
    curl \
    gnupg2 \
    lsb-release \
    locales && \
    locale-gen en_US.UTF-8

# Set locale
ENV LANG=en_US.UTF-8 \
    LANGUAGE=en_US:en \
    LC_ALL=en_US.UTF-8

# Install ROS Noetic
RUN echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list && \
    curl -sSL http://repo.ros2.org/repos.key | apt-key add - && \
    apt-get update && \
    apt-get install -y ros-noetic-desktop-full && \
    echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc && \
    /bin/bash -c "source /opt/ros/noetic/setup.bash"

# Install TensorBoard and PyTorch
RUN pip3 install --no-cache-dir \
    tensorboard \
    torch==1.10.0 \
    torchvision==0.11.1 \
    torchaudio==0.10.0

# Create workspace
WORKDIR /catkin_ws
RUN mkdir -p src

# Setup environment variables
ENV ROS_HOSTNAME=localhost \
    ROS_MASTER_URI=http://localhost:11311 \
    ROS_PORT_SIM=11311 \
    GAZEBO_RESOURCE_PATH=~/DRL-robot-navigation/catkin_ws/src/multi_robot_scenario/launch

RUN pip3 install squaternion    

RUN apt-get install -y \ 
    libgl1-mesa-glx \ 
    libgl1-mesa-dri \ 
    mesa-utils \
    x11-xserver-utils

# Expose necessary ports
EXPOSE 11311 6006

# Entry point
CMD ["/bin/bash"]
