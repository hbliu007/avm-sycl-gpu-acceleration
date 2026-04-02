# SYCL GPU Acceleration Integration for AV2 Codec

## Summary

This document summarizes the integration of SYCL GPU acceleration into the AV2 video codec with RTCD (Run-Time CPU Detection) support.

## Files Modified

### 1. CMake Build System

#### `av2/av2.cmake`
- Added SYCL source files to `AVM_AV2_COMMON_SOURCES`:
  - `av2/common/av2_inv_txfm_sycl.c` - Inverse transform SYCL wrappers
- Added SYCL source files to `AVM_AV2_ENCODER_SOURCES`:
  - `av2/encoder/av2_fwd_txfm_sycl.c` - Forward transform SYCL wrappers
- Added SYCL-specific compiler flags and library linking in `setup_av2_targets()`

#### `avm_dsp/avm_dsp.cmake`
- Added SYCL subdirectory inclusion and library linking in `setup_avm_dsp_targets()`
- Links `avm_sycl` library to main `avm` target when `HAVE_SYCL` is enabled

### 2. RTCD Configuration

#### `av2/common/av2_rtcd_defs.pl`
- Added `sycl` specialization to transform functions:
  - `inv_stxfm`: Added SYCL specialization
  - `av2_highbd_inv_txfm_add`: Added SYCL specialization
  - `fwd_stxfm`: Added SYCL specialization
  - `fwd_txfm`: Added SYCL specialization

### 3. SYCL Source Files

#### Core SYCL Implementation (`avm_dsp/sycl/`)
- `sycl_context.cpp/hpp` - SYCL context and device management
- `sycl_txfm.cpp/hpp` - Transform kernels (DCT, ADST, IDTX)
- `sycl_txfm_optimized.cpp` - Optimized transform kernels
- `sycl_me.cpp/hpp` - Motion estimation kernels
- `sycl_lpf.cpp/hpp` - Loop filter kernels
- `sycl_intra.cpp/hpp` - Intra prediction kernels
- `sycl_api.cpp/h` - High-level SYCL API
- `sycl_wrapper.cpp/hpp` - C wrapper functions for SYCL

#### AV2 Integration Files
- `av2/common/av2_inv_txfm_sycl.c`
  - C wrapper functions for inverse transforms
  - Handles GPU memory allocation and data transfer
  - Falls back to C implementation when SYCL is unavailable

- `av2/encoder/av2_fwd_txfm_sycl.c`
  - C wrapper functions for forward transforms
  - Handles GPU memory allocation and data transfer
  - Falls back to C implementation when SYCL is unavailable

## Build Configuration

### Enable SYCL Support

To enable SYCL GPU acceleration, configure CMake with:

```bash
cmake -DAVM_ENABLE_SYCL=ON -DSYCL_BACKEND=CUDA ..
```

Or for Intel/Metal backends:

```bash
cmake -DAVM_ENABLE_SYCL=ON -DSYCL_BACKEND=METAL ..
```

### Compiler Flags

SYCL sources are compiled with:
- `-fsycl` - Enable SYCL support
- `-fsycl-targets=nvptx64-nvidia-cuda` - For CUDA backend
- `-fsycl-targets=spir64_gen` - For Intel/Metal backend

## RTCD Integration

The RTCD system automatically selects the best implementation at runtime:

1. **CPU SIMD**: AVX2, SSE4.1, SSSE3, NEON (selected based on CPU features)
2. **GPU SYCL**: Automatically used when GPU is available and SYCL is enabled
3. **C Fallback**: Used when no optimized implementation is available

Function selection priority: GPU SYCL > CPU SIMD > C

## Transform Support

### Supported Transform Sizes
- 4x4, 8x8, 16x16, 32x32, 64x64

### Supported Transform Types
- DCT-II (Discrete Cosine Transform Type II)
- ADST (Asymmetric Discrete Sine Transform)
- IDTX (Identity Transform)
- Hybrid transforms (different row/column types)

### Memory Management
- Device buffers are allocated per transform call
- Data is transferred between host and device as needed
- Automatic fallback to C implementation for unsupported sizes

## Testing

To verify SYCL integration:

1. Build with SYCL enabled:
   ```bash
   mkdir build && cd build
   cmake -DAVM_ENABLE_SYCL=ON -DCONFIG_AV2_ENCODER=ON ..
   make
   ```

2. Run the minimal SYCL test:
   ```bash
   cd /tmp/avm-sycl-gpu-acceleration
   icpx -fsycl minimal_sycl_test.cpp -o minimal_sycl_test
   ONEAPI_DEVICE_SELECTOR=cuda:0 ./minimal_sycl_test
   ```

3. Check for SYCL device detection in application output:
   ```
   [SYCL] Initialized successfully
     Device: <GPU Name>
     Backend: <Backend Name>
     Type: GPU
   ```

## Performance Considerations

### GPU Memory Transfer Overhead
- For small blocks (4x4, 8x8), CPU SIMD may be faster due to memory transfer overhead
- GPU acceleration is most beneficial for larger blocks (16x16, 32x32, 64x64)

### Batch Processing
- The current implementation processes individual blocks
- Future optimizations could batch multiple blocks for better GPU utilization

### Async Operations
- Current implementation uses synchronous waits for simplicity
- Future versions could use async operations and streams for better performance

## Future Enhancements

1. **Batch Processing**: Process multiple blocks in a single kernel launch
2. **Pinned Memory**: Use pinned host memory for faster transfers
3. **Async Streams**: Overlap computation and data transfer
4. **More Transforms**: Add GPU kernels for more transform types
5. **Motion Estimation**: Full GPU acceleration for motion search
6. **Loop Filters**: GPU acceleration for CDEF, GDF, restoration filters

## Troubleshooting

### SYCL Not Detected
- Ensure Intel oneAPI or compatible SYCL compiler is installed
- Check that `-fsycl` flag is supported by your compiler
- Verify `AVM_ENABLE_SYCL=ON` is set in CMake

### GPU Not Found
- Check GPU drivers are installed and working
- Verify `ONEAPI_DEVICE_SELECTOR` environment variable
- Check SYCL backend selection (CUDA vs METAL vs OpenCL)

### Build Errors
- Ensure SYCL headers are in include path
- Check for conflicts between SYCL and other SIMD flags
- Verify all SYCL sources are compiled with `-fsycl`

## References

- Intel oneAPI Documentation: https://www.intel.com/content/www/us/en/developer/tools/oneapi/overview.html
- SYCL Specification: https://www.khronos.org/sycl/
- AV2 Codec Documentation: See AOM repository
