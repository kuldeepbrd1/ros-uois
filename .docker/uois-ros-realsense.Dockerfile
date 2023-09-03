FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

# ENV
ENV HOME_DIR=/root/
ENV WS_DIR=${HOME_DIR}/workspaces
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV ROS_DISTRO noetic
ENV INSTALL_CUSTOM_CERT=false
ENV CERT_FILE=""

# REQUIREMENTS & CERTS
# Copy requirements and any private certificates from '.docker'
COPY .docker/*.crt /usr/local/share/ca-certificates/
SHELL ["/bin/bash", "-c"]

# Set env var INSTALL_CUSTOM_CERT if a custom certificate is to be installed
# hotfix- cuda source error on ubuntu 20.04
RUN if ls /usr/local/share/ca-certificates/*.crt >/dev/null 2>&1; then \
    echo "INSTALL_CUSTOM_CERT=true" >> /etc/environment; \
    export CERT_FILE=$(ls /usr/local/share/ca-certificates/*.crt | head -n 1); \
    fi

RUN rm /etc/apt/sources.list.d/cuda.list
# RUN echo "deb [by-hash=no] http://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64 /" > /etc/apt/sources.list.d/cuda.list

# APT
RUN apt-get update -y\
    && apt-get upgrade -y \
    && DEBIAN_FRONTEND=noninteractive \
    apt-get install -q -y --no-install-recommends \
    build-essential \
    curl \
    cmake \
    dirmngr \
    gnupg2 \
    git \
    iputils-ping \
    ca-certificates \
    libegl1-mesa-dev \
    libgl1-mesa-dev \
    libgles2-mesa-dev \
    libglvnd-dev  \
    libglu1-mesa \
    libsm6 \
    libxi6 \
    libxrandr2 \
    libxt6 \
    vulkan-tools \
    nano \
    net-tools \
    python3-dev \
    python3-pip \
    python3-wheel \
    python3-opengl \
    qtbase5-dev \
    libqt5core5a \
    libqt5gui5 \
    libqt5widgets5 \
    tree \
    unzip \
    wget \
    && rm -rf /var/lib/apt/lists/* \
    && update-ca-certificates \
    && echo "alias python=python3" >> /root/.bashrc\
    && echo "alias pip=pip3" >> /root/.bashrc

# ROS
RUN echo "deb http://packages.ros.org/ros/ubuntu focal main" > /etc/apt/sources.list.d/ros1-latest.list \
    && apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654 \
    && apt-get update && apt-get install -y --no-install-recommends \
    ros-${ROS_DISTRO}-ros-core=1.5.0-1* \
    python3-rospy \
    python3-rosdep \
    python3-rosinstall \
    python3-catkin-tools \
    ros-${ROS_DISTRO}-realsense2-camera \
    && rosdep init \
    && rm -rf /var/lib/apt/lists/*

# Configure Bashrc convenience
RUN echo "WS_DIR=${WS_DIR}" >> ~/.bashrc \
    && echo 'source /opt/ros/${ROS_DISTRO}/setup.bash' >> ~/.bashrc \
    && echo '[ -f "${WS_DIR}/devel/setup.bash" ] && source "${WS_DIR}/devel/setup.bash"' >> ~/.bashrc \
    && echo 'alias source-devel="source ${WS_DIR}/devel/setup.bash"' >> ~/.bashrc

# Global CA Cert config, if custom cert file exists
RUN if [ "$INSTALL_CUSTOM_CERT" = "true" ]; then \
    cp ${CERT_FILE} '$(openssl version -d | cut -f2 -d \")/certs' \
    && cat ${CERT_FILE} >> $(python -m certifi) \
    && echo "export CERT_PATH=$(python3 -m certifi)" >> ~/.bashrc \
    && echo "export SSL_CERT_FILE=${CERT_PATH}"  >> ~/.bashrc \
    && echo "export REQUESTS_CA_BUNDLE=${CERT_PATH}"  >> ~/.bashrc;  \
    fi

## NOTE: Separating COPY/RUN commands into several layers
# saves time in rebuilding during development
# TO optimize image size, they can be combined into one layer

# ROS-GRASP Python dependencies
COPY ./requirements.txt ${WS_DIR}/src/requirements.txt
RUN pip install -r ${WS_DIR}/src/requirements.txt && rm ${WS_DIR}/src/requirements.txt

# UOIS
COPY ./uois ${WS_DIR}/src/ros_uois/uois

RUN cd ${WS_DIR}/src/ros_uois/uois \
    && pip install -r requirements.txt \
    && pip install -e .

# SAM
COPY ./segment-anything ${WS_DIR}/src/ros_uois/segment-anything

RUN cd ${WS_DIR}/src/ros_uois/segment-anything \
    && pip install -e .

# Package
COPY action ${WS_DIR}/src/ros_uois/action
COPY msg ${WS_DIR}/src/ros_uois/msg
COPY scripts ${WS_DIR}/src/ros_uois/scripts
COPY ros_uois ${WS_DIR}/src/ros_uois/ros_uois
COPY package.xml ${WS_DIR}/src/ros_uois/package.xml
COPY CMakeLists.txt ${WS_DIR}/src/ros_uois/CMakeLists.txt
COPY setup.py ${WS_DIR}/src/ros_uois/setup.py


# Install ROS dependencies/packages
RUN apt-get update  \
    && source /opt/ros/${ROS_DISTRO}/setup.bash \
    && cd ${WS_DIR} \
    && rosdep update \
    && rosdep install -y -r -i --rosdistro "${ROS_DISTRO}" --from-paths "${WS_DIR}/src" \
    && catkin build

CMD [ "/bin/bash" ]
