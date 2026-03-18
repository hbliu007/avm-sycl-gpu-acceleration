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

#ifndef AVM_AVM_DSP_SYCL_SYCL_LPF_HPP_
#define AVM_AVM_DSP_SYCL_SYCL_LPF_HPP_

#include <sycl/sycl.hpp>
#include "avm_dsp/avm_dsp_common.h"
#include "avm_dsp/loopfilter.h"

namespace avm {
namespace sycl {

// Loop filter constants
constexpr int DF_SHIFT = 8;
constexpr int DF_8_THRESH = 3;
constexpr int DF_6_THRESH = 4;
constexpr int FILT_8_THRESH_SHIFT = 3;
constexpr int DF_Q_THRESH_SHIFT = 4;
constexpr int MAX_DBL_FLT_LEN = 8;
constexpr int SEC_DERIV_ARRAY_LEN = (MAX_DBL_FLT_LEN + 1) * 2;

// Filter width multipliers
inline constexpr int w_mult[MAX_DBL_FLT_LEN] = { 85, 51, 37, 28, 23, 20, 17, 15 };

// Q threshold multipliers
inline constexpr int q_thresh_mults[MAX_DBL_FLT_LEN] = { 32, 25, 19, 19,
                                                          18, 18, 17, 17 };

// Q threshold for double filter decisions
inline constexpr int q_first[5] = { 45, 43, 40, 35, 32 };

// Loop filter parameters structure
struct LpfParams {
  int filt_width_neg;   // Filter width on negative side
  int filt_width_pos;   // Filter width on positive side
  uint16_t q_thresh;    // Quality threshold
  uint16_t side_thresh; // Side threshold
  int bd;               // Bit depth (8, 10, or 12)
  int is_lossless_neg;  // Lossless mode on negative side
  int is_lossless_pos;  // Lossless mode on positive side
};

// ============================================================================
// Horizontal Loop Filter Kernels
// ============================================================================

/// @brief Apply horizontal loop filter with width 4 (LPF_HORIZONTAL_4)
/// @param q SYCL queue for kernel execution
/// @param s Source/destination pixel buffer (pointer to edge)
/// @param pitch Stride between rows in pixels
/// @param params Filter parameters
/// @param count Number of pixels to filter (typically 4 or 8)
void lpf_horizontal_4(::sycl::queue& q, uint16_t* s, int pitch,
                      const LpfParams& params, int count = 4);

/// @brief Apply horizontal loop filter with width 8 (LPF_HORIZONTAL_8)
/// @param q SYCL queue for kernel execution
/// @param s Source/destination pixel buffer (pointer to edge)
/// @param pitch Stride between rows in pixels
/// @param params Filter parameters
/// @param count Number of pixels to filter
void lpf_horizontal_8(::sycl::queue& q, uint16_t* s, int pitch,
                      const LpfParams& params, int count = 8);

/// @brief Apply horizontal loop filter with width 14 (LPF_HORIZONTAL_14)
/// @param q SYCL queue for kernel execution
/// @param s Source/destination pixel buffer (pointer to edge)
/// @param pitch Stride between rows in pixels
/// @param params Filter parameters
/// @param count Number of pixels to filter
void lpf_horizontal_14(::sycl::queue& q, uint16_t* s, int pitch,
                       const LpfParams& params, int count = 4);

// ============================================================================
// Vertical Loop Filter Kernels
// ============================================================================

/// @brief Apply vertical loop filter with width 4 (LPF_VERTICAL_4)
/// @param q SYCL queue for kernel execution
/// @param s Source/destination pixel buffer (pointer to edge)
/// @param pitch Stride between rows in pixels
/// @param params Filter parameters
/// @param count Number of pixels to filter (typically 4 or 8)
void lpf_vertical_4(::sycl::queue& q, uint16_t* s, int pitch,
                    const LpfParams& params, int count = 4);

/// @brief Apply vertical loop filter with width 8 (LPF_VERTICAL_8)
/// @param q SYCL queue for kernel execution
/// @param s Source/destination pixel buffer (pointer to edge)
/// @param pitch Stride between rows in pixels
/// @param params Filter parameters
/// @param count Number of pixels to filter
void lpf_vertical_8(::sycl::queue& q, uint16_t* s, int pitch,
                    const LpfParams& params, int count = 8);

/// @brief Apply vertical loop filter with width 14 (LPF_VERTICAL_14)
/// @param q SYCL queue for kernel execution
/// @param s Source/destination pixel buffer (pointer to edge)
/// @param pitch Stride between rows in pixels
/// @param params Filter parameters
/// @param count Number of pixels to filter
void lpf_vertical_14(::sycl::queue& q, uint16_t* s, int pitch,
                     const LpfParams& params, int count = 4);

// ============================================================================
// Dual Filter Variants (Horizontal + Vertical)
// ============================================================================

/// @brief Apply both horizontal and vertical loop filter at a corner
/// @param q SYCL queue for kernel execution
/// @param s Source/destination pixel buffer (pointer to corner)
/// @param pitch Stride between rows in pixels
/// @param params_h Horizontal filter parameters
/// @param params_v Vertical filter parameters
void lpf_dual(::sycl::queue& q, uint16_t* s, int pitch,
              const LpfParams& params_h, const LpfParams& params_v);

/// @brief Apply horizontal and vertical filters for a block (width 8)
/// @param q SYCL queue for kernel execution
/// @param s Source/destination pixel buffer
/// @param pitch Stride between rows in pixels
/// @param params Filter parameters
/// @param width Block width in pixels
/// @param height Block height in pixels
void lpf_horizontal_edge_8(::sycl::queue& q, uint16_t* s, int pitch,
                           const LpfParams& params, int width, int height);

/// @brief Apply vertical and horizontal filters for a block (width 8)
/// @param q SYCL queue for kernel execution
/// @param s Source/destination pixel buffer
/// @param pitch Stride between rows in pixels
/// @param params Filter parameters
/// @param width Block width in pixels
/// @param height Block height in pixels
void lpf_vertical_edge_8(::sycl::queue& q, uint16_t* s, int pitch,
                         const LpfParams& params, int width, int height);

// ============================================================================
// Filter Selection and Decision Functions
// ============================================================================

/// @brief Determine the number of samples to modify for current row/column
/// @param q SYCL queue for kernel execution
/// @param s Pixel buffer at the edge
/// @param pitch Stride between rows (1 for vertical)
/// @param max_filt_neg Maximum filter width on negative side
/// @param max_filt_pos Maximum filter width on positive side
/// @param q_thresh Quality threshold
/// @param side_thresh Side threshold
/// @param t Corresponding pixels in adjacent block
/// @return Number of samples to filter (0 to MAX_DBL_FLT_LEN)
int filt_choice(::sycl::queue& q, uint16_t* s, int pitch,
                int max_filt_neg, int max_filt_pos,
                uint16_t q_thresh, uint16_t side_thresh, uint16_t* t);

// ============================================================================
// Batch Processing Functions
// ============================================================================

/// @brief Apply horizontal filter to multiple edges in parallel
/// @param q SYCL queue for kernel execution
/// @param edges Array of edge pointers
/// @param pitch Stride between rows in pixels
/// @param params Array of filter parameters for each edge
/// @param num_edges Number of edges to process
/// @param width Filter width (4, 8, or 14)
void batch_lpf_horizontal(::sycl::queue& q, uint16_t** edges, int pitch,
                          const LpfParams* params, int num_edges, int width);

/// @brief Apply vertical filter to multiple edges in parallel
/// @param q SYCL queue for kernel execution
/// @param edges Array of edge pointers
/// @param pitch Stride between rows in pixels
/// @param params Array of filter parameters for each edge
/// @param num_edges Number of edges to process
/// @param width Filter width (4, 8, or 14)
void batch_lpf_vertical(::sycl::queue& q, uint16_t** edges, int pitch,
                        const LpfParams* params, int num_edges, int width);

// ============================================================================
// Utility Functions
// ============================================================================

/// @brief Clamp pixel value to valid range for given bit depth
/// @param value Input value
/// @param bd Bit depth (8, 10, or 12)
/// @return Clamped value
inline uint16_t clamp_pixel(int value, int bd) {
  const int max = (1 << bd) - 1;
  return static_cast<uint16_t>(sycl::clamp(value, 0, max));
}

/// @brief Apply asymmetric filter operation
/// @param s Pixel buffer at edge
/// @param pitch Stride (1 for vertical, row stride for horizontal)
/// @param delta_m2 Delta value to apply
/// @param width Filter width
/// @param q_thresh_clamp Clamped quality threshold
/// @param bd Bit depth
/// @param is_lossless Lossless mode flag
void apply_filter_asym(uint16_t* s, int pitch, int delta_m2,
                       int width, int q_thresh_clamp, int bd,
                       int is_lossless_neg, int is_lossless_pos);

/// @brief Compute second derivative for filter decision
/// @param s Pixel buffer
/// @param t Corresponding pixels in adjacent block
/// @param pitch Stride
/// @param offset Offset from edge
/// @return Second derivative value
inline int compute_second_deriv(const uint16_t* s, const uint16_t* t,
                                int pitch, int offset) {
  int deriv_s = abs(s[offset * pitch] - (s[0] << 1) + s[-offset * pitch]);
  int deriv_t = abs(t[offset * pitch] - (t[0] << 1) + t[-offset * pitch]);
  return (deriv_s + deriv_t + 1) >> 1;
}

}  // namespace sycl
}  // namespace avm

#endif  // AVM_AVM_DSP_SYCL_SYCL_LPF_HPP_
