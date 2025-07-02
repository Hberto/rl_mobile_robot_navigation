FROM ros:humble-desktop

ENV DEBIAN_FRONTEND=noninteractive

# Update and install essential dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-colcon-common-extensions \
    ros-humble-gazebo-ros-pkgs \
    wget \
    git \
    build-essential \
    cmake \
    libboost-all-dev \
    curl \
    gnupg2 \
    xterm \
    tmux \
    dbus-x11 \
    xdg-utils \
    lsb-release \
    && rm -rf /var/lib/apt/lists/*



# Install TensorBoard and PyTorch and other Pip packages
RUN pip3 install --no-cache-dir \
    squaternion \
    numpy \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 \
    tensorboard

# Create workspace
WORKDIR /ros2_ws

RUN . /opt/ros/humble/setup.sh && \
    rosdep init && \
    rosdep update

# Entry point
CMD ["/bin/bash"]
