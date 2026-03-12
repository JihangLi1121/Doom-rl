FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libboost-all-dev \
    libbz2-dev \
    libfluidsynth-dev \
    libfreetype6-dev \
    libgme-dev \
    libgtk2.0-dev \
    libjpeg-dev \
    libopenal-dev \
    libpng-dev \
    libsdl2-dev \
    libwildmidi-dev \
    libzmq3-dev \
    nano \
    nasm \
    pkg-config \
    rsync \
    software-properties-common \
    sudo \
    tar \
    timidity \
    unzip \
    wget \
    zlib1g-dev \
    python3.10 \
    python3.10-dev \
    python3-pip \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

RUN pip install --upgrade pip setuptools wheel

RUN pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

RUN pip install tensorflow==2.16.1

RUN pip install vizdoom gymnasium

RUN pip install \
    stable-baselines3[extra]==2.2.1 \
    sb3-contrib==2.2.1 \
    tensorboard==2.15.1 \
    wandb==0.16.0 \
    opencv-python==4.8.1.78 \
    imageio==2.33.0 \
    imageio-ffmpeg==0.4.9

RUN pip install \
    numpy==1.24.3 \
    pandas==2.1.4 \
    matplotlib==3.8.2 \
    seaborn==0.13.0 \
    scikit-learn==1.3.2 \
    scipy==1.11.4 \
    pillow==10.1.0 \
    tqdm==4.66.1 \
    jupyter==1.0.0 \
    jupyterlab==4.0.9 \
    ipython==8.18.1

RUN pip install \
    gym-notices==0.0.8 \
    ale-py==0.8.1 \
    pettingzoo==1.24.3 \
    supersuit==3.9.2

WORKDIR /workspace

RUN jupyter notebook --generate-config && \
    echo "c.NotebookApp.ip = '0.0.0.0'" >> ~/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.open_browser = False" >> ~/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.allow_root = True" >> ~/.jupyter/jupyter_notebook_config.py

EXPOSE 6006 8888

CMD ["/bin/bash"]
