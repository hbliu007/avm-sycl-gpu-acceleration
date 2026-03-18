/*
 * Copyright (c) 2021, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 3-Clause Clear License
 * and the Alliance for Open Media Patent License 1.0. If the BSD 3-Clause Clear
 * License was not distributed with this source code in the LICENSE file, you
 * can obtain it at aomedia.org/license/software-license/bsd-3-c-c/.  If the
 * Alliance for Open Media Patent License 1.0 was not distributed with this
 * source code in the PATENTS file, you can obtain it at
 * aomedia.org/license/patent-license/.
 */

#ifndef AVM_AVM_DSP_SYCL_SYCL_ME_HPP_
#define AVM_AVM_DSP_SYCL_SYCL_ME_HPP_

#include <sycl/sycl.hpp>
#include <cstdint>
#include "av2/common/enums.h"

namespace avm {
namespace sycl {

// ============================================================================
// Motion Estimation Configuration
// ============================================================================

/// @brief Motion estimation search parameters
struct MEParams {
  int search_range;      // Search window size in pixels
  int start_x;           // Starting MV x component (1/8 pixel units)
  int start_y;           // Starting MV y component (1/8 pixel units)
  BLOCK_SIZE bsize;      // Block size for motion estimation
  int bd;                // Bit depth (8, 10, or 12)
  bool use_highbd;       // Whether using high bit depth
};

/// @brief Motion vector result
struct MVResult {
  int mv_x;              // Best MV x component (1/8 pixel units)
  int mv_y;              // Best MV y component (1/8 pixel units)
  uint32_t sad;          // SAD value at best position
};

/// @brief SAD computation result with position
struct SADResult {
  int x;                 // X offset from search center
  int y;                 // Y offset from search center
  uint32_t sad;          // SAD value
};

// ============================================================================
// SAD Kernels - Basic Block Sizes
// ============================================================================

/// @brief Compute 4x4 SAD between source and reference blocks
/// @param q SYCL queue for execution
/// @param src Source block (current frame)
/// @param src_stride Source stride in pixels
/// @param ref Reference block (reference frame, search window)
/// @param ref_stride Reference stride in pixels
/// @param ref_x Starting X position in reference
/// @param ref_y Starting Y position in reference
/// @param results Output array of SAD values [search_range * search_range]
/// @param params Motion estimation parameters
void sad4x4(::sycl::queue& q, const uint16_t* src, int src_stride,
            const uint16_t* ref, int ref_stride, int ref_x, int ref_y,
            uint32_t* results, const MEParams& params);

/// @brief Compute 8x8 SAD between source and reference blocks
void sad8x8(::sycl::queue& q, const uint16_t* src, int src_stride,
            const uint16_t* ref, int ref_stride, int ref_x, int ref_y,
            uint32_t* results, const MEParams& params);

/// @brief Compute 16x16 SAD between source and reference blocks
void sad16x16(::sycl::queue& q, const uint16_t* src, int src_stride,
              const uint16_t* ref, int ref_stride, int ref_x, int ref_y,
              uint32_t* results, const MEParams& params);

/// @brief Compute 32x32 SAD between source and reference blocks
void sad32x32(::sycl::queue& q, const uint16_t* src, int src_stride,
              const uint16_t* ref, int ref_stride, int ref_x, int ref_y,
              uint32_t* results, const MEParams& params);

/// @brief Compute 64x64 SAD between source and reference blocks
void sad64x64(::sycl::queue& q, const uint16_t* src, int src_stride,
              const uint16_t* ref, int ref_stride, int ref_x, int ref_y,
              uint32_t* results, const MEParams& params);

// ============================================================================
// SAD Kernels - Rectangular Blocks
// ============================================================================

/// @brief Compute 4x8 SAD
void sad4x8(::sycl::queue& q, const uint16_t* src, int src_stride,
            const uint16_t* ref, int ref_stride, int ref_x, int ref_y,
            uint32_t* results, const MEParams& params);

/// @brief Compute 8x4 SAD
void sad8x4(::sycl::queue& q, const uint16_t* src, int src_stride,
            const uint16_t* ref, int ref_stride, int ref_x, int ref_y,
            uint32_t* results, const MEParams& params);

/// @brief Compute 8x16 SAD
void sad8x16(::sycl::queue& q, const uint16_t* src, int src_stride,
             const uint16_t* ref, int ref_stride, int ref_x, int ref_y,
             uint32_t* results, const MEParams& params);

/// @brief Compute 16x8 SAD
void sad16x8(::sycl::queue& q, const uint16_t* src, int src_stride,
             const uint16_t* ref, int ref_stride, int ref_x, int ref_y,
             uint32_t* results, const MEParams& params);

/// @brief Compute 16x32 SAD
void sad16x32(::sycl::queue& q, const uint16_t* src, int src_stride,
              const uint16_t* ref, int ref_stride, int ref_x, int ref_y,
              uint32_t* results, const MEParams& params);

/// @brief Compute 32x16 SAD
void sad32x16(::sycl::queue& q, const uint16_t* src, int src_stride,
              const uint16_t* ref, int ref_stride, int ref_x, int ref_y,
              uint32_t* results, const MEParams& params);

// ============================================================================
// Multi-Candidate SAD (for diamond search, hexagonal search, etc.)
// ============================================================================

/// @brief Compute SAD for multiple candidate positions simultaneously
/// @param q SYCL queue for execution
/// @param src Source block
/// @param src_stride Source stride
/// @param ref Reference frame window
/// @param ref_stride Reference stride
/// @param candidates Array of candidate positions (x, y offsets)
/// @param num_candidates Number of candidates
/// @param results Output SAD values [num_candidates]
/// @param width Block width
/// @param height Block height
void sad_multi_candidate(::sycl::queue& q, const uint16_t* src, int src_stride,
                         const uint16_t* ref, int ref_stride,
                         const int2* candidates, int num_candidates,
                         uint32_t* results, int width, int height);

/// @brief 4-direction SAD for diamond search refinement
/// @param q SYCL queue for execution
/// @param src Source block
/// @param src_stride Source stride
/// @param ref Reference frame
/// @param ref_stride Reference stride
/// @param center_x Current search center X
/// @param center_y Current search center Y
/// @param sad_out Output SAD values [4] for: (0,-1), (0,1), (-1,0), (1,0)
/// @param width Block width
/// @param height Block height
void sad_diamond_4way(::sycl::queue& q, const uint16_t* src, int src_stride,
                      const uint16_t* ref, int ref_stride,
                      int center_x, int center_y, uint32_t* sad_out,
                      int width, int height);

/// @brief 8-point SAD for diamond/octagonal search
/// @param q SYCL queue for execution
/// @param src Source block
/// @param src_stride Source stride
/// @param ref Reference frame
/// @param ref_stride Reference stride
/// @param center_x Current search center X
/// @param center_y Current search center Y
/// @param sad_out Output SAD values [8] for surrounding positions
/// @param width Block width
/// @param height Block height
void sad_diamond_8way(::sycl::queue& q, const uint16_t* src, int src_stride,
                      const uint16_t* ref, int ref_stride,
                      int center_x, int center_y, uint32_t* sad_out,
                      int width, int height);

// ============================================================================
// Full Search Motion Estimation
// ============================================================================

/// @brief Full search motion estimation for a block
/// @param q SYCL queue for execution
/// @param src Source block (current frame)
/// @param src_stride Source stride in pixels
/// @param ref Reference frame
/// @param ref_stride Reference stride in pixels
/// @param ref_x Reference block origin X
/// @param ref_y Reference block origin Y
/// @param params Motion estimation parameters
/// @return Best motion vector with minimum SAD
MVResult full_search_me(::sycl::queue& q, const uint16_t* src, int src_stride,
                        const uint16_t* ref, int ref_stride,
                        int ref_x, int ref_y, const MEParams& params);

/// @brief Diamond search motion estimation (iterative refinement)
/// @param q SYCL queue for execution
/// @param src Source block
/// @param src_stride Source stride
/// @param ref Reference frame
/// @param ref_stride Reference stride
/// @param start_mv Starting motion vector (1/8 pixel units)
/// @param params Motion estimation parameters
/// @return Refined motion vector
MVResult diamond_search_me(::sycl::queue& q, const uint16_t* src, int src_stride,
                           const uint16_t* ref, int ref_stride,
                           const MVResult& start_mv, const MEParams& params);

// ============================================================================
// Hierarchical Motion Estimation
// ============================================================================

/// @brief Hierarchical ME using downsampled frames
/// @param q SYCL queue for execution
/// @param src Original resolution source
/// @param src_stride Source stride
/// @param ref Original resolution reference
/// @param ref_stride Reference stride
/// @param src_ds 2x downsampled source
/// @param src_ds_stride Downsampled source stride
/// @param ref_ds 2x downsampled reference
/// @param ref_ds_stride Downsampled reference stride
/// @param params Motion estimation parameters
/// @return Best motion vector at full resolution
MVResult hierarchical_me(::sycl::queue& q,
                        const uint16_t* src, int src_stride,
                        const uint16_t* ref, int ref_stride,
                        const uint16_t* src_ds, int src_ds_stride,
                        const uint16_t* ref_ds, int ref_ds_stride,
                        const MEParams& params);

// ============================================================================
// Sub-pixel Motion Estimation
// ============================================================================

/// @brief Half-pixel refinement using interpolation
/// @param q SYCL queue for execution
/// @param src Source block
/// @param src_stride Source stride
/// @param ref Interpolated reference (with half-pel positions)
/// @param ref_stride Reference stride
/// @param start_mv Integer-pixel MV to refine
/// @param params Motion estimation parameters
/// @return Refined motion vector with half-pel precision
MVResult subpel_halfpel_me(::sycl::queue& q, const uint16_t* src, int src_stride,
                           const uint16_t* ref, int ref_stride,
                           const MVResult& start_mv, const MEParams& params);

/// @brief Quarter-pixel refinement
/// @param q SYCL queue for execution
/// @param src Source block
/// @param src_stride Source stride
/// @param ref Interpolated reference (with quarter-pel positions)
/// @param ref_stride Reference stride
/// @param start_mv Half-pixel MV to refine
/// @param params Motion estimation parameters
/// @return Refined motion vector with quarter-pel precision
MVResult subpel_quarterpel_me(::sycl::queue& q, const uint16_t* src, int src_stride,
                              const uint16_t* ref, int ref_stride,
                              const MVResult& start_mv, const MEParams& params);

// ============================================================================
// Batch Motion Estimation (for multiple blocks)
// ============================================================================

/// @brief Motion estimation for multiple blocks in parallel
/// @param q SYCL queue for execution
/// @param src Source frame
/// @param src_stride Source stride
/// @param ref Reference frame
/// @param ref_stride Reference stride
/// @param block origins Array of block origins (x, y)
/// @param num_blocks Number of blocks to process
/// @param params Motion estimation parameters (same for all blocks)
/// @param results Output MV results [num_blocks]
void batch_full_search_me(::sycl::queue& q,
                          const uint16_t* src, int src_stride,
                          const uint16_t* ref, int ref_stride,
                          const int2* block_origins, int num_blocks,
                          const MEParams& params, MVResult* results);

// ============================================================================
// Utility Functions
// ============================================================================

/// @brief Get SAD function pointer for specific block size
/// @param bsize Block size
/// @return Function pointer to SAD kernel
using SADFunc = void(*)(::sycl::queue&, const uint16_t*, int,
                        const uint16_t*, int, int, int,
                        uint32_t*, const MEParams&);

SADFunc get_sad_function(BLOCK_SIZE bsize);

/// @brief Compute block dimensions from BLOCK_SIZE
/// @param bsize Block size enum
/// @param width Output width
/// @param height Output height
void get_block_dimensions(BLOCK_SIZE bsize, int& width, int& height);

/// @brief Convert full-pel MV to sub-pel precision
/// @param mv_x Full-pel MV x
/// @param mv_y Full-pel MV y
/// @param precision Target precision (1=full, 2=half, 4=quarter, 8=eighth)
inline void mv_to_subpel(int& mv_x, int& mv_y, int precision) {
  mv_x *= precision;
  mv_y *= precision;
}

/// @brief Clip motion vector to valid range
/// @param mv_x MV x component
/// @param mv_y MV y component
/// @param x_min Minimum x (full-pel units)
/// @param x_max Maximum x (full-pel units)
/// @param y_min Minimum y (full-pel units)
/// @param y_max Maximum y (full-pel units)
inline void clip_mv(int& mv_x, int& mv_y,
                    int x_min, int x_max, int y_min, int y_max) {
  mv_x = sycl::clamp(mv_x, x_min, x_max);
  mv_y = sycl::clamp(mv_y, y_min, y_max);
}

/// @brief SAD computation device function (for use in other kernels)
/// @param src_local Source block in local memory
/// @param ref_local Reference block in local memory
/// @param width Block width
/// @param height Block height
/// @param local_stride Local memory stride
/// @return SAD value
inline uint32_t compute_sad_local(const uint16_t* src_local,
                                  const uint16_t* ref_local,
                                  int width, int height, int local_stride) {
  uint32_t sad = 0;
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      sad += sycl::abs(src_local[y * local_stride + x] -
                       ref_local[y * local_stride + x]);
    }
  }
  return sad;
}

}  // namespace sycl
}  // namespace avm

#endif  // AVM_AVM_DSP_SYCL_SYCL_ME_HPP_
