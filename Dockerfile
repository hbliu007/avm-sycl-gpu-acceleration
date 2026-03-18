# AVM SYCL GPU Acceleration - Docker Image
# Multi-stage build for minimal image size

# Build stage
FROM intel/oneapi-hpckit:latest AS builder

WORKDIR /build

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    cmake \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Copy source
COPY . .

# Build
RUN mkdir build && cd build && \
    cmake .. -GNinja \
             -DCMAKE_CXX_COMPILER=icpx \
             -DCMAKE_BUILD_TYPE=Release \
             -DAVM_ENABLE_SYCL=ON \
             -DAVM_BUILD_TESTS=ON \
             -DAVM_BUILD_EXAMPLES=ON && \
    ninja

# Test
RUN cd build && ctest --output-on-failure

# Runtime stage
FROM intel/oneapi-runtime:latest

WORKDIR /app

# Copy binaries from builder
COPY --from=builder /build/build/tests/sycl_context_test /app/bin/
COPY --from=builder /build/build/tests/sycl_txfm_test /app/bin/
COPY --from=builder /build/build/tests/sycl_perf_test /app/bin/
COPY --from=builder /build/build/examples/basic_example /app/bin/
COPY --from=builder /build/src/*.hpp /app/include/

# Copy documentation
COPY README.md /app/
COPY docs/ /app/docs/

# Set up entry point
ENV LD_LIBRARY_PATH=/opt/intel/oneapi/lib:$LD_LIBRARY_PATH

# Default command
CMD ["/app/bin/sycl_context_test"]

# Labels
LABEL org.opencontainers.image.title="AVM SYCL GPU Acceleration"
LABEL org.opencontainers.image.description="Cross-platform SYCL GPU acceleration for AV2 codec"
LABEL org.opencontainers.image.url="https://github.com/hbliu007/avm-sycl-gpu-acceleration"
LABEL org.opencontainers.image.source="https://github.com/hbliu007/avm-sycl-gpu-acceleration"
LABEL org.opencontainers.image.vendor="hbliu007"

# Usage:
# docker build -t avm-sycl .
# docker run --gpus all -it avm-sycl
# docker run --gpus all -it avm-sycl /app/bin/sycl_perf_test --benchmark
