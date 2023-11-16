# Use an official Ubuntu as a parent image
FROM ubuntu:20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTORCH_VERSION=2.1.0

# Install required packages and add deadsnakes PPA
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y build-essential cmake python3.10 python3.10-dev python3.10-distutils wget git libopenmpi-dev openmpi-bin

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
RUN wget https://bootstrap.pypa.io/get-pip.py && \
	python3 get-pip.py && \
	python3 -m pip install --upgrade pip

# Download and install PyTorch dependencies
RUN pip install numpy pyyaml mkl mkl-include setuptools cmake cffi typing ninja -U

# Clone the PyTorch source code from GitHub
RUN git clone --branch v${PYTORCH_VERSION} --recursive https://github.com/pytorch/pytorch

# Build PyTorch from source
ENV USE_MPI=1
ENV USE_CUDA=0
RUN cd /pytorch && \
	python3 setup.py build develop

# Install PyTorch
RUN cd /pytorch && \
	python3 setup.py install

# Clean up
RUN apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

COPY requirements-cpu.txt requirements.txt
RUN pip install -r requirements.txt