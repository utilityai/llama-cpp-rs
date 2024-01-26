ARG CUDA_VERSION=12.3.1
ARG UBUNTU_VERSION=22.04
FROM nvcr.io/nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} as base-cuda
RUN DEBIAN_FRONTEND=noninteractive apt update -y && apt install -y curl
RUN curl https://sh.rustup.rs -sSf | bash -s -- -y
ENV PATH=/root/.cargo/bin:$PATH
# Install requirements for bindgen: https://rust-lang.github.io/rust-bindgen/requirements.html
RUN apt update && apt install -y llvm-dev libclang-dev clang
COPY . .
RUN cargo build --package llama-cpp-sys-2 --features cublas