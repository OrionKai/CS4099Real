# Build stage: compile Rust app
FROM --platform=$BUILDPLATFORM rust:1.82 AS build

ARG TARGETARCH
ARG TARGETVARIANT

# Install cross-compilation tool based on target architecture

# Determine Rust target based on target architecture
RUN case "$TARGETARCH" in \
    "amd64")  export RUST_TARGET="x86_64-unknown-linux-gnu" ;; \
    "arm64")  export RUST_TARGET="aarch64-unknown-linux-gnu" ;; \
    "arm")    export RUST_TARGET="armv7-unknown-linux-gnueabihf" ;; \
    *)        echo "Unsupported architecture: $TARGETARCH" && exit 1 ;; \
    esac && \
    rustup target add $RUST_TARGET


WORKDIR /app

COPY src ./src
COPY Cargo.toml Cargo.lock ./

RUN cargo build --release --target $RUST_TARGET

# Final stage: create runtime image
FROM ubuntu:22.04
COPY script/ .

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTORCH_VERSION=2.5.1
ENV PYTORCH_ABI=libtorch-cxx11-abi

RUN apt-get update && apt-get install -y \
    curl \
    unzip \
    ca-certificates \
    libgomp1 \
    binutils \
    && rm -rf /var/lib/apt/lists/* \
    && ldconfig

RUN export PYTORCH_VERSION="2.5.1"
RUN export PYTORCH_ABI="libtorch-cxx11-abi"

RUN curl -s -L -O --remote-name-all https://download.pytorch.org/libtorch/cpu/${PYTORCH_ABI}-shared-with-deps-${PYTORCH_VERSION}%2Bcpu.zip \
    && unzip -q "${PYTORCH_ABI}-shared-with-deps-${PYTORCH_VERSION}%2Bcpu.zip" \
    && rm -f "${PYTORCH_ABI}-shared-with-deps-${PYTORCH_VERSION}%2Bcpu.zip"

ENV LD_LIBRARY_PATH=/libtorch/lib

RUN chmod u+x execute.sh

# CMD ["scripts/execute.sh"]