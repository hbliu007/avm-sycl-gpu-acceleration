# SYCL GPU Acceleration Integration Guide

This guide explains how to integrate SYCL GPU acceleration into your AV2 video encoding workflow.

## Overview

The SYCL module provides GPU-accelerated implementations of compute-intensive AV2 codec operations:

- **DCT/IDCT Transforms** - 8x8, 16x16, 32x32 forward and inverse transforms
- **Motion Estimation** - SAD for various block sizes (4x4 to 64x64)
- **Loop Filtering** - Deblocking filter operations
- **Intra Prediction** - DC, H, V, smooth, and directional modes

## Integration Methods

### 1. Direct API Usage

The simplest way to use SYCL acceleration:

```cpp
#include "sycl_wrapper.hpp"

int main() {
    // Initialize SYCL context (auto GPU selection)
    auto& ctx = avm::sycl::SYCLContext::instance();
    if (!ctx.initialize()) {
        std::cerr << "SYCL initialization failed" << std::endl;
        return 1;
    }

    // Use GPU-accelerated DCT
    int16_t input[64] = {...};
    int32_t output[64];
    avm::sycl::fdct8x8(ctx.queue(), input, output);

    return 0;
}
```

### 2. RTCD Integration

For seamless CPU/GPU fallback, integrate with AV2's Runtime CPU Dispatch (RTCD):

```cpp
// In your rtcd.c or rtcd_impl.cpp

#if HAVE_SYCL
#include "sycl/sycl_wrapper.hpp"

// SYCL-accelerated function with CPU fallback
void avm_fdct8x8(const int16_t *input, tran_low_t *output, int stride) {
    if (avm::sycl::should_use_sycl()) {
        auto& ctx = avm::sycl::SYCLContext::instance();
        avm::sycl::fdct8x8(ctx.queue(), input, output, stride);
    } else {
        // Fallback to CPU implementation
        avm_fdct8x8_c(input, output, stride);
    }
}
#endif
```

### 3. Conditional Compilation

Add SYCL support to your CMakeLists.txt:

```cmake
# Check for SYCL support
include(cmake/sycl.cmake)

if(AVM_HAVE_SYCL)
    add_definitions(-DHAVE_SYCL=1)
    add_subdirectory(sycl)
    target_link_libraries(avm PRIVATE avm_sycl)
endif()
```

## FFmpeg Integration

To use SYCL acceleration with FFmpeg:

```cpp
// In libavcodec/av1dec.c or similar

#if CONFIG_SYCL
#include "sycl/sycl_wrapper.hpp"

static int decode_frame_with_sycl(AVCodecContext *avctx, AVFrame *frame) {
    // Initialize SYCL on first use
    static bool sycl_initialized = false;
    if (!sycl_initialized) {
        auto& ctx = avm::sycl::SYCLContext::instance();
        ctx.initialize();
        sycl_initialized = true;
    }

    // Use SYCL-accelerated functions
    // ...
}
#endif
```

### FFmpeg Build Configuration

```bash
# Build FFmpeg with SYCL support
./configure --enable-libaom --enable-sycl \
    --extra-cflags="-fsycl" \
    --extra-ldflags="-fsycl"
make -j$(nproc)
```

## OpenCV Integration

Use SYCL acceleration in OpenCV video processing:

```cpp
#include <opencv2/opencv.hpp>
#include "sycl_wrapper.hpp"

class SYCLAcceleratedEncoder {
public:
    SYCLAcceleratedEncoder() {
        auto& ctx = avm::sycl::SYCLContext::instance();
        ctx.initialize();
    }

    void processFrame(const cv::Mat& frame) {
        // Convert to AV2-compatible format
        // Apply SYCL-accelerated transforms
        auto& ctx = avm::sycl::SYCLContext::instance();

        // DCT example
        int16_t block[64];
        int32_t dct[64];
        // ... fill block from frame ...
        avm::sycl::fdct8x8(ctx.queue(), block, dct);
    }
};
```

## GStreamer Integration

Create a GStreamer element with SYCL acceleration:

```c
// gstav1syclenc.c

#include <gst/gst.h>
#include "sycl_wrapper.hpp"

GST_DEBUG_CATEGORY_STATIC(gst_av1_sycl_enc_debug);
#define GST_CAT_DEFAULT gst_av1_sycl_enc_debug

static gboolean
gst_av1_sycl_enc_init (GstAV1SyclEnc * self)
{
    auto& ctx = avm::sycl::SYCLContext::instance();
    if (!ctx.initialize()) {
        GST_WARNING_OBJECT(self, "SYCL initialization failed, using CPU");
        self->use_sycl = FALSE;
    } else {
        self->use_sycl = TRUE;
        GST_INFO_OBJECT(self, "SYCL initialized: %s",
                        ctx.device_name().c_str());
    }
    return TRUE;
}
```

## Performance Considerations

### When to Use GPU vs CPU

| Operation Count | Recommended Path |
|----------------|------------------|
| < 1,000 ops | CPU (SIMD) |
| 1,000 - 10,000 ops | GPU (SYCL) |
| > 10,000 ops | GPU (SYCL) + Batching |

### Batch Processing

For maximum performance, batch multiple operations:

```cpp
// Process multiple DCT blocks in parallel
std::vector<int16_t*> input_blocks(num_blocks);
std::vector<int32_t*> output_blocks(num_blocks);

auto& ctx = avm::sycl::SYCLContext::instance();
for (int i = 0; i < num_blocks; ++i) {
    avm::sycl::fdct8x8(ctx.queue(), input_blocks[i], output_blocks[i]);
}
ctx.queue().wait();  // Single synchronization point
```

### Memory Management

SYCL uses Unified Shared Memory (USM) for zero-copy transfers:

```cpp
// Allocate USM memory for frequent transfers
auto& ctx = avm::sycl::SYCLContext::instance();
int16_t* device_mem = ::sycl::malloc_device<int16_t>(64, ctx.queue());

// Use directly in kernels
// ...

// Free when done
::sycl::free(device_mem, ctx.queue());
```

## Error Handling

Always check for SYCL availability:

```cpp
#include "sycl_wrapper.hpp"

void encode_frame(Frame* frame) {
    if (!avm::sycl::is_available()) {
        // Use CPU path
        encode_frame_cpu(frame);
        return;
    }

    try {
        auto& ctx = avm::sycl::SYCLContext::instance();
        // Use GPU path
        encode_frame_sycl(ctx, frame);
    } catch (const ::sycl::exception& e) {
        // Fallback to CPU on SYCL errors
        std::cerr << "SYCL error: " << e.what() << std::endl;
        encode_frame_cpu(frame);
    }
}
```

## Troubleshooting

### No GPU Devices Found

```
[SYCL] No devices found
```

**Solution:** Install the appropriate SYCL backend:
- NVIDIA GPUs: Install AdaptiveCpp or Intel DPC++ with CUDA support
- Intel GPUs: Install Intel oneAPI with Level Zero
- AMD GPUs: Install AdaptiveCpp with ROCm backend

### Kernel Compilation Errors

```
error: kernel compilation failed
```

**Solution:** Check SYCL compiler version and target:
```bash
icpx -fsycl -fsycl-targets=nvptx64-nvidia-cuda your_code.cpp
```

### Performance Issues

If GPU performance is slower than CPU:
1. Check data transfer overhead
2. Increase batch size
3. Verify GPU is actually being used: `SYCL_BE=PI_CUDA your_program`

## API Reference

### SYCLContext

```cpp
class SYCLContext {
public:
    static SYCLContext& instance();  // Singleton access

    bool initialize();               // Initialize GPU
    bool is_available() const;       // Check if GPU available
    bool is_gpu() const;             // Check if using GPU (vs CPU fallback)

    ::sycl::queue& queue();          // Get SYCL queue
    const std::string& backend_name() const;  // Backend name
    size_t compute_units() const;    // Number of compute units
    size_t global_mem_size() const;  // Global memory size

    static std::vector<PlatformInfo> list_devices();  // List all devices
};
```

### Transform Functions

```cpp
namespace avm::sycl {

void fdct8x8(::sycl::queue& q, const int16_t* input, int32_t* output);
void idct8x8(::sycl::queue& q, const int32_t* input, int16_t* output);
void fdct16x16(::sycl::queue& q, const int16_t* input, int32_t* output);
void idct16x16(::sycl::queue& q, const int32_t* input, int16_t* output);
void fdct32x32(::sycl::queue& q, const int16_t* input, int32_t* output);
void idct32x32(::sycl::queue& q, const int32_t* input, int16_t* output);

}
```

### Motion Estimation Functions

```cpp
namespace avm::sycl {

uint32_t sad4x4(::sycl::queue& q, const uint16_t* src, const uint16_t* ref);
uint32_t sad8x8(::sycl::queue& q, const uint16_t* src, const uint16_t* ref);
uint32_t sad16x16(::sycl::queue& q, const uint16_t* src, const uint16_t* ref);
uint32_t sad32x32(::sycl::queue& q, const uint16_t* src, const uint16_t* ref);
uint32_t sad64x64(::sycl::queue& q, const uint16_t* src, const uint16_t* ref);

}
```
