FROM ubuntu:20.04
COPY execute.sh .

RUN apt-get update && apt-get install -y \
    curl \
    unzip \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTORCH_VERSION=2.4.1
ENV PYTORCH_ABI=libtorch-cxx11-abi

RUN export PYTORCH_VERSION="2.4.1"
RUN export PYTORCH_ABI="libtorch-cxx11-abi"

RUN curl -s -L -O --remote-name-all https://download.pytorch.org/libtorch/cpu/${PYTORCH_ABI}-shared-with-deps-${PYTORCH_VERSION}%2Bcpu.zip \
    && unzip -q "${PYTORCH_ABI}-shared-with-deps-${PYTORCH_VERSION}%2Bcpu.zip" \
    && rm -f "${PYTORCH_ABI}-shared-with-deps-${PYTORCH_VERSION}%2Bcpu.zip"

ENV LD_LIBRARY_PATH=/libtorch/lib:${LD_LIBRARY_PATH}

# CMD ["scripts/execute.sh"]