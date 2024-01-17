# Builds the project in a docker container This is used to test arm and x86 builds using github actions + docker.
# This also requires us to declare all the dependencies in the dockerfile usful as documentation.
FROM rust:bookworm AS builder
# Install requirements for bindgen: https://rust-lang.github.io/rust-bindgen/requirements.html
RUN apt update && apt install -y llvm-dev libclang-dev clang
COPY . .
RUN cargo build --release --example simple

FROM debian:bookworm-slim
COPY --from=builder /target/release/examples/simple /usr/local/bin/simple
ENTRYPOINT ["/usr/local/bin/simple"]