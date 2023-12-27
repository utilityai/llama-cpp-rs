ARG CUDA_VERSION=12.2.0
ARG UBUNTU_VERSION=22.04

FROM nvcr.io/nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} as base-cuda

RUN apt update -y

FROM base-cuda as rust-cuda

RUN DEBIAN_FRONTEND=noninteractive \
    apt install -y \
    curl \
    libssl-dev \
    libclang-dev \
    pkg-config \
    cmake \
    git \
    protobuf-compiler
RUN curl https://sh.rustup.rs -sSf | bash -s -- -y
ENV PATH=/root/.cargo/bin:$PATH

RUN cargo install cargo-chef --locked

FROM rust-cuda as planner
WORKDIR /app
COPY . .
RUN cargo chef prepare --recipe-path recipe.json

FROM rust-cuda as deps-built-cudnn
WORKDIR /app
COPY --from=planner /app/recipe.json recipe.json
RUN cargo chef cook --release --recipe-path recipe.json
COPY . .
RUN cargo build --lib --release --features=cublas

FROM deps-built-cudnn as built-cudnn-cli
WORKDIR /app
COPY --from=deps-built-cudnn /app/. .
ARG CUDA_DOCKER_ARCH=all
ENV CUDA_DOCKER_ARCH=${CUDA_DOCKER_ARCH}
RUN cargo build --bin llama-cpp-cli --release --features=cublas

FROM deps-built-cudnn as built-cudnn-rpc
WORKDIR /app
COPY --from=deps-built-cudnn /app/. .
ARG CUDA_DOCKER_ARCH=all
ENV CUDA_DOCKER_ARCH=${CUDA_DOCKER_ARCH}
RUN cargo build --bin llama-cpp-rpc --release --features=cublas

FROM deps-built-cudnn as built-cudnn-server
WORKDIR /app
COPY --from=deps-built-cudnn /app/. .
ARG CUDA_DOCKER_ARCH=all
ENV CUDA_DOCKER_ARCH=${CUDA_DOCKER_ARCH}
RUN cargo build --bin llama-cpp-server --release --features=cublas

FROM nvcr.io/nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} as cli

WORKDIR /app

COPY --from=built-cudnn-cli /app/target/release/llama-cpp-cli llama-cpp-cli

ENTRYPOINT ["./llama-cpp-cli"]

FROM nvcr.io/nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} as rpc

WORKDIR /app

COPY --from=built-cudnn-rpc /app/target/release/llama-cpp-rpc llama-cpp-rpc

ENTRYPOINT ["./llama-cpp-rpc"]

FROM nvcr.io/nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} as server

WORKDIR /app

COPY --from=built-cudnn-server /app/target/release/llama-cpp-server llama-cpp-server

ENTRYPOINT ["./llama-cpp-server"]