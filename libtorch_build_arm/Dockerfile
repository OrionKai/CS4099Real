FROM ubuntu:20.04

# Install dependencies
RUN apt-get update && \
    apt-get install -y \
        build-essential \
        cmake \
        git \
        curl \
        libopenblas-dev \
        libomp-dev \
        && rm -rf /var/lib/apt/lists/*

# Set up environment variables
ENV CMAKE_PREFIX_PATH=/usr/local/libtorch

# Clone PyTorch repository
RUN git clone --recursive https://github.com/pytorch/pytorch.git /pytorch

WORKDIR /pytorch

# Checkout a stable release
RUN git checkout v1.13.1

# Build `libtorch`
RUN python3 tools/build_libtorch.py
