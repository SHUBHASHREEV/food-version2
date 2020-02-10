ARG PYTORCH="1.1.0"
ARG CUDA="10.0"
ARG CUDNN="7.5"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

RUN apt-get update && apt-get install -y libglib2.0-0 libsm6 libxrender-dev libxext6 \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    build-essential \
    bzip2 \
    cmake \
    curl \
    git \
    g++ \
    libboost-all-dev \
    pkg-config \
    rsync \
    software-properties-common \
    sudo \
    tar \
    timidity \
    unzip \
    wget \
    locales \
    zlib1g-dev \
    python3-dev \
    python3 \
    python3-pip \
    python3-tk \
    libjpeg-dev \
    libpng-dev

# Python3
RUN pip3 install pip --upgrade
RUN pip3 install utm cython aicrowd_api timeout_decorator \
  numpy \
  aicrowd-repo2docker \
  pillow
RUN pip3 install git+https://github.com/AIcrowd/coco.git#subdirectory=PythonAPI
	
RUN conda install cython -y && conda clean --all
RUN git clone --branch v1.0rc1 https://github.com/open-mmlab/mmdetection.git /mmdetection
WORKDIR /mmdetection
RUN pip install --no-cache-dir -e .
RUN python3.6 -m pip install aicrowd_api   aicrowd-repo2docker
# Unicode support:
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

# Enables X11 sharing and creates user home directory
ENV USER_NAME aicrowd
ENV HOME_DIR /home/$USER_NAME
#
# Replace HOST_UID/HOST_GUID with your user / group id (needed for X11)
ENV HOST_UID 1000
ENV HOST_GID 1000

RUN export uid=${HOST_UID} gid=${HOST_GID} && \
    mkdir -p ${HOME_DIR} && \
    echo "$USER_NAME:x:${uid}:${gid}:$USER_NAME,,,:$HOME_DIR:/bin/bash" >> /etc/passwd && \
    echo "$USER_NAME:x:${uid}:" >> /etc/group && \
    echo "$USER_NAME ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/$USER_NAME && \
    chmod 0440 /etc/sudoers.d/$USER_NAME && \
    chown ${uid}:${gid} -R ${HOME_DIR}

USER ${USER_NAME}
WORKDIR ${HOME_DIR}

COPY . .

RUN sudo chown ${HOST_UID}:${HOST_GID} -R *
RUN sudo chmod 775 -R *


