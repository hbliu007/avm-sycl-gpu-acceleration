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

#ifdef HAVE_SYCL

#include "sycl_txfm.hpp"
#include "sycl_context.hpp"
#include "avm_dsp_common.h"

#include <sycl/sycl.hpp>
#include <cmath>
#include <algorithm>

namespace avm {
namespace sycl {

// ============================================================================
// DCT Coefficient Tables (Fixed-point, scaled for 8-bit precision)
// ============================================================================

// DCT-II coefficients for 8-point transform
// Values are scaled by 2^14 (16384) for fixed-point arithmetic
// cos(i * pi / 16) for i = 1..7
namespace {

// 8-point DCT-II butterfly coefficients (scaled by 2^14)
// These are cos(pi/16), cos(2*pi/16), ..., cos(7*pi/16)
constexpr int kDct8CosPi16 = 16069;   // cos(pi/16) * 16384
constexpr int kDct8Cos2Pi16 = 15137;  // cos(2*pi/16) * 16384
constexpr int kDct8Cos3Pi16 = 13623;  // cos(3*pi/16) * 16384
constexpr int kDct8Cos4Pi16 = 11585;  // cos(4*pi/16) * 16384
constexpr int kDct8Cos5Pi16 = 9102;   // cos(5*pi/16) * 16384
constexpr int kDct8Cos6Pi16 = 6270;   // cos(6*pi/16) * 16384
constexpr int kDct8Cos7Pi16 = 3196;   // cos(7*pi/16) * 16384

// 8-point DCT rotation coefficients (scaled by 2^14)
constexpr int kDct8SinPi8 = 8867;     // sin(pi/8) * 16384
constexpr int kDct8CosPi8 = 15137;    // cos(pi/8) * 16384
constexpr int kDct8Sin3Pi8 = 13623;   // sin(3*pi/8) * 16384
constexpr int kDct8Cos3Pi8 = 6270;    // cos(3*pi/8) * 16384

// Shift amounts for different transform stages
constexpr int kFwdShift8x8[] = {2, 1, 1};   // Forward transform shifts
constexpr int kInvShift8x8[] = {1, 2, 1};   // Inverse transform shifts

// 4-point DCT coefficients
constexpr int kDct4CosPi8 = 15137;    // cos(pi/8) * 16384
constexpr int kDct4SinPi8 = 8867;     // sin(pi/8) * 16384
constexpr int kDct4Cos3Pi8 = 6270;    // cos(3*pi/8) * 16384
constexpr int kDct4Sin3Pi8 = 13623;   // sin(3*pi/8) * 16384

// Round and shift helper for SYCL kernel
inline tran_low_t round_shift_kernel(tran_high_t value, int shift) {
  if (shift <= 0) return static_cast<tran_low_t>(value);
  return static_cast<tran_low_t>((value + (tran_high_t{1} << (shift - 1))) >> shift);
}

// Clamp pixel value for given bit depth
inline uint16_t clamp_pixel_kernel(int value, int bd) {
  const int max_val = (1 << bd) - 1;
  return static_cast<uint16_t>(sycl::clamp(value, 0, max_val));
}

// ============================================================================
// 8-point DCT Row/Column Transform Kernels
// ============================================================================

// Forward 8-point DCT kernel (1D)
// Implements Chen's algorithm for DCT-II
void fdct8_kernel(tran_low_t* row) {
  // Stage 1: Even/odd separation and butterfly
  tran_low_t step[8];

  // Even part: 0, 2, 4, 6
  tran_low_t s0 = row[0] + row[7];
  tran_low_t s1 = row[1] + row[6];
  tran_low_t s2 = row[2] + row[5];
  tran_low_t s3 = row[3] + row[4];

  tran_low_t s4 = row[0] - row[7];
  tran_low_t s5 = row[1] - row[6];
  tran_low_t s6 = row[2] - row[5];
  tran_low_t s7 = row[3] - row[4];

  // Even part butterfly
  tran_low_t even0 = s0 + s3;
  tran_low_t even1 = s1 + s2;
  tran_low_t even2 = s1 - s2;
  tran_low_t even3 = s0 - s3;

  // 4-point DCT on even part
  step[0] = even0 + even1;
  step[4] = even0 - even1;

  // Rotation for even2, even3
  step[2] = static_cast<tran_low_t>((kDct8Cos2Pi16 * even2 + kDct8Cos6Pi16 * even3) >> 14);
  step[6] = static_cast<tran_low_t>((kDct8Cos6Pi16 * even2 - kDct8Cos2Pi16 * even3) >> 14);

  // Odd part: 1, 3, 5, 7 using rotation
  step[1] = static_cast<tran_low_t>((kDct8CosPi16 * s4 + kDct8Cos7Pi16 * s7) >> 14);
  step[7] = static_cast<tran_low_t>((kDct8Cos7Pi16 * s4 - kDct8CosPi16 * s7) >> 14);
  step[3] = static_cast<tran_low_t>((kDct8Cos3Pi16 * s5 + kDct8Cos5Pi16 * s6) >> 14);
  step[5] = static_cast<tran_low_t>((kDct8Cos5Pi16 * s5 - kDct8Cos3Pi16 * s6) >> 14);

  // Final butterfly for odd part
  tran_low_t odd0 = step[1] + step[3];
  tran_low_t odd1 = step[1] - step[3];
  tran_low_t odd2 = step[5] - step[7];
  tran_low_t odd3 = step[5] + step[7];

  row[0] = step[0];
  row[1] = odd0;
  row[2] = step[2];
  row[3] = odd2;
  row[4] = step[4];
  row[5] = odd3;
  row[6] = step[6];
  row[7] = odd1;
}

// Inverse 8-point DCT kernel (1D)
void idct8_kernel(tran_low_t* row) {
  tran_low_t step[8];

  // Stage 1: Reconstruction from output of fdct8
  tran_low_t odd0 = row[1];
  tran_low_t odd1 = row[7];
  tran_low_t odd2 = row[3];
  tran_low_t odd3 = row[5];

  // Inverse odd butterfly
  step[1] = odd0 + odd1;
  step[3] = odd0 - odd1;
  step[5] = odd3 - odd2;
  step[7] = odd3 + odd2;

  // Inverse rotation
  tran_low_t s4 = static_cast<tran_low_t>((kDct8CosPi16 * step[1] - kDct8Cos7Pi16 * step[7]) >> 14);
  tran_low_t s7 = static_cast<tran_low_t>((kDct8Cos7Pi16 * step[1] + kDct8CosPi16 * step[7]) >> 14);
  tran_low_t s5 = static_cast<tran_low_t>((kDct8Cos3Pi16 * step[3] - kDct8Cos5Pi16 * step[5]) >> 14);
  tran_low_t s6 = static_cast<tran_low_t>((kDct8Cos5Pi16 * step[3] + kDct8Cos3Pi16 * step[5]) >> 14);

  // Even part
  tran_low_t even0 = row[0];
  tran_low_t even1 = row[4];
  tran_low_t even2 = row[2];
  tran_low_t even3 = row[6];

  tran_low_t s0 = even0 + even1;
  tran_low_t s3 = even0 - even1;

  // Inverse rotation for even part
  tran_low_t s1 = static_cast<tran_low_t>((kDct8Cos2Pi16 * even2 - kDct8Cos6Pi16 * even3) >> 14);
  tran_low_t s2 = static_cast<tran_low_t>((kDct8Cos6Pi16 * even2 + kDct8Cos2Pi16 * even3) >> 14);

  // Final even/odd combination
  row[0] = s0 + s1;
  row[1] = s4 + s5;
  row[2] = s2 + s3;
  row[3] = s6 + s7;
  row[4] = s3 - s2;
  row[5] = s7 - s6;
  row[6] = s1 - s0;
  row[7] = s5 - s4;
}

// ============================================================================
// 4-point DCT Row/Column Transform Kernels
// ============================================================================

void fdct4_kernel(tran_low_t* row) {
  // Even part
  tran_low_t s0 = row[0] + row[3];
  tran_low_t s1 = row[1] + row[2];
  tran_low_t s2 = row[0] - row[3];
  tran_low_t s3 = row[1] - row[2];

  tran_low_t even0 = s0 + s1;
  tran_low_t even1 = s0 - s1;

  // Odd part with rotation
  tran_low_t odd0 = static_cast<tran_low_t>((kDct4CosPi8 * s2 + kDct4SinPi8 * s3) >> 14);
  tran_low_t odd1 = static_cast<tran_low_t>((kDct4SinPi8 * s2 - kDct4CosPi8 * s3) >> 14);

  row[0] = even0;
  row[1] = odd0;
  row[2] = even1;
  row[3] = odd1;
}

void idct4_kernel(tran_low_t* row) {
  tran_low_t even0 = row[0] + row[2];
  tran_low_t even1 = row[0] - row[2];

  tran_low_t odd0 = row[1];
  tran_low_t odd1 = row[3];

  // Inverse rotation
  tran_low_t s2 = static_cast<tran_low_t>((kDct4CosPi8 * odd0 - kDct4SinPi8 * odd1) >> 14);
  tran_low_t s3 = static_cast<tran_low_t>((kDct4SinPi8 * odd0 + kDct4CosPi8 * odd1) >> 14);

  tran_low_t s0 = even0;
  tran_low_t s1 = even1;

  row[0] = s0 + s2;
  row[1] = s1 + s3;
  row[2] = s1 - s3;
  row[3] = s0 - s2;
}

}  // anonymous namespace

// ============================================================================
// Forward 8x8 DCT Implementation
// ============================================================================

void fdct8x8(::sycl::queue& q, const int16_t* input, tran_low_t* output,
             int stride, const TxfmParams& params) {
  constexpr int kSize = 8;
  const int bd = params.bd;

  // Allocate USM memory for device computation
  int16_t* input_dev = sycl::malloc_shared<int16_t>(kSize * stride, q);
  tran_low_t* temp_dev = sycl::malloc_shared<tran_low_t>(kSize * kSize, q);
  tran_low_t* output_dev = sycl::malloc_shared<tran_low_t>(kSize * kSize, q);

  // Copy input to device
  q.memcpy(input_dev, input, kSize * stride * sizeof(int16_t)).wait();

  // Stage 1: Row transform (read from input, write to temp)
  q.submit([&](::sycl::handler& cgh) {
    cgh.parallel_for(::sycl::range<2>(kSize, kSize), [=](::sycl::id<2> idx) {
      const int row = idx[0];
      const int col = idx[1];

      // Load row into local buffer with stride handling
      tran_low_t row_buf[kSize];
      for (int c = 0; c < kSize; ++c) {
        row_buf[c] = static_cast<tran_low_t>(input_dev[row * stride + c]);
      }

      // Apply first shift
      for (int c = 0; c < kSize; ++c) {
        row_buf[c] = row_buf[c] << kFwdShift8x8[0];
      }

      // Forward DCT on row
      fdct8_kernel(row_buf);

      // Apply second shift and store
      for (int c = 0; c < kSize; ++c) {
        temp_dev[row * kSize + c] = round_shift_kernel(row_buf[c], kFwdShift8x8[1]);
      }
    });
  }).wait();

  // Stage 2: Column transform (read from temp, write to output)
  q.submit([&](::sycl::handler& cgh) {
    cgh.parallel_for(::sycl::range<2>(kSize, kSize), [=](::sycl::id<2> idx) {
      const int col = idx[1];

      // Only process once per column (using first row as anchor)
      if (idx[0] != 0) return;

      // Load column into local buffer
      tran_low_t col_buf[kSize];
      for (int r = 0; r < kSize; ++r) {
        col_buf[r] = temp_dev[r * kSize + col];
      }

      // Forward DCT on column
      fdct8_kernel(col_buf);

      // Apply third shift and store
      for (int r = 0; r < kSize; ++r) {
        output_dev[r * kSize + col] = round_shift_kernel(col_buf[r], kFwdShift8x8[2]);
      }
    });
  }).wait();

  // Copy output back
  q.memcpy(output, output_dev, kSize * kSize * sizeof(tran_low_t)).wait();

  // Free device memory
  ::sycl::free(input_dev, q);
  ::sycl::free(temp_dev, q);
  ::sycl::free(output_dev, q);
}

// ============================================================================
// Inverse 8x8 DCT Implementation
// ============================================================================

void idct8x8(::sycl::queue& q, const tran_low_t* input, uint16_t* output,
             int stride, const TxfmParams& params) {
  constexpr int kSize = 8;
  const int bd = params.bd;

  // Allocate USM memory
  tran_low_t* input_dev = sycl::malloc_shared<tran_low_t>(kSize * kSize, q);
  tran_low_t* temp_dev = sycl::malloc_shared<tran_low_t>(kSize * kSize, q);
  uint16_t* output_dev = sycl::malloc_shared<uint16_t>(kSize * stride, q);

  // Initialize output buffer
  q.memset(output_dev, 0, kSize * stride * sizeof(uint16_t)).wait();

  // Copy input to device
  q.memcpy(input_dev, input, kSize * kSize * sizeof(tran_low_t)).wait();

  // Stage 1: Column transform (inverse)
  q.submit([&](::sycl::handler& cgh) {
    cgh.parallel_for(::sycl::range<2>(kSize, kSize), [=](::sycl::id<2> idx) {
      const int col = idx[1];

      if (idx[0] != 0) return;

      // Load column
      tran_low_t col_buf[kSize];
      for (int r = 0; r < kSize; ++r) {
        col_buf[r] = input_dev[r * kSize + col];
      }

      // Inverse DCT on column
      idct8_kernel(col_buf);

      // Store with first inverse shift
      for (int r = 0; r < kSize; ++r) {
        temp_dev[r * kSize + col] = round_shift_kernel(col_buf[r], kInvShift8x8[0]);
      }
    });
  }).wait();

  // Stage 2: Row transform (inverse) and final output
  q.submit([&](::sycl::handler& cgh) {
    cgh.parallel_for(::sycl::range<2>(kSize, kSize), [=](::sycl::id<2> idx) {
      const int row = idx[0];
      const int col = idx[1];

      // Each work item handles one row's transform
      if (col != 0) return;

      // Load row
      tran_low_t row_buf[kSize];
      for (int c = 0; c < kSize; ++c) {
        row_buf[c] = temp_dev[row * kSize + c];
      }

      // Inverse DCT on row
      idct8_kernel(row_buf);

      // Apply remaining shifts and clamp to pixel range
      for (int c = 0; c < kSize; ++c) {
        int val = round_shift_kernel(row_buf[c], kInvShift8x8[1] + kInvShift8x8[2]);
        output_dev[row * stride + c] = clamp_pixel_kernel(val, bd);
      }
    });
  }).wait();

  // Copy output back
  q.memcpy(output, output_dev, kSize * stride * sizeof(uint16_t)).wait();

  // Free device memory
  ::sycl::free(input_dev, q);
  ::sycl::free(temp_dev, q);
  ::sycl::free(output_dev, q);
}

// ============================================================================
// Forward 4x4 DCT Implementation
// ============================================================================

void fdct4x4(::sycl::queue& q, const int16_t* input, tran_low_t* output,
             int stride, const TxfmParams& params) {
  constexpr int kSize = 4;
  const int bd = params.bd;

  int16_t* input_dev = sycl::malloc_shared<int16_t>(kSize * stride, q);
  tran_low_t* temp_dev = sycl::malloc_shared<tran_low_t>(kSize * kSize, q);
  tran_low_t* output_dev = sycl::malloc_shared<tran_low_t>(kSize * kSize, q);

  q.memcpy(input_dev, input, kSize * stride * sizeof(int16_t)).wait();

  // Row transform
  q.submit([&](::sycl::handler& cgh) {
    cgh.parallel_for(::sycl::range<2>(kSize, kSize), [=](::sycl::id<2> idx) {
      const int row = idx[0];
      const int col = idx[1];

      tran_low_t row_buf[kSize];
      for (int c = 0; c < kSize; ++c) {
        row_buf[c] = static_cast<tran_low_t>(input_dev[row * stride + c]);
      }

      fdct4_kernel(row_buf);

      for (int c = 0; c < kSize; ++c) {
        temp_dev[row * kSize + c] = row_buf[c];
      }
    });
  }).wait();

  // Column transform
  q.submit([&](::sycl::handler& cgh) {
    cgh.parallel_for(::sycl::range<2>(kSize, kSize), [=](::sycl::id<2> idx) {
      const int col = idx[1];

      if (idx[0] != 0) return;

      tran_low_t col_buf[kSize];
      for (int r = 0; r < kSize; ++r) {
        col_buf[r] = temp_dev[r * kSize + col];
      }

      fdct4_kernel(col_buf);

      for (int r = 0; r < kSize; ++r) {
        output_dev[r * kSize + col] = round_shift_kernel(col_buf[r], 1);
      }
    });
  }).wait();

  q.memcpy(output, output_dev, kSize * kSize * sizeof(tran_low_t)).wait();

  ::sycl::free(input_dev, q);
  ::sycl::free(temp_dev, q);
  ::sycl::free(output_dev, q);
}

// ============================================================================
// Inverse 4x4 DCT Implementation
// ============================================================================

void idct4x4(::sycl::queue& q, const tran_low_t* input, uint16_t* output,
             int stride, const TxfmParams& params) {
  constexpr int kSize = 4;
  const int bd = params.bd;

  tran_low_t* input_dev = sycl::malloc_shared<tran_low_t>(kSize * kSize, q);
  tran_low_t* temp_dev = sycl::malloc_shared<tran_low_t>(kSize * kSize, q);
  uint16_t* output_dev = sycl::malloc_shared<uint16_t>(kSize * stride, q);

  q.memset(output_dev, 0, kSize * stride * sizeof(uint16_t)).wait();
  q.memcpy(input_dev, input, kSize * kSize * sizeof(tran_low_t)).wait();

  // Column transform
  q.submit([&](::sycl::handler& cgh) {
    cgh.parallel_for(::sycl::range<2>(kSize, kSize), [=](::sycl::id<2> idx) {
      const int col = idx[1];

      if (idx[0] != 0) return;

      tran_low_t col_buf[kSize];
      for (int r = 0; r < kSize; ++r) {
        col_buf[r] = input_dev[r * kSize + col];
      }

      idct4_kernel(col_buf);

      for (int r = 0; r < kSize; ++r) {
        temp_dev[r * kSize + col] = col_buf[r];
      }
    });
  }).wait();

  // Row transform and output
  q.submit([&](::sycl::handler& cgh) {
    cgh.parallel_for(::sycl::range<2>(kSize, kSize), [=](::sycl::id<2> idx) {
      const int row = idx[0];
      const int col = idx[1];

      if (col != 0) return;

      tran_low_t row_buf[kSize];
      for (int c = 0; c < kSize; ++c) {
        row_buf[c] = temp_dev[row * kSize + c];
      }

      idct4_kernel(row_buf);

      for (int c = 0; c < kSize; ++c) {
        int val = round_shift_kernel(row_buf[c], 1);
        output_dev[row * stride + c] = clamp_pixel_kernel(val, bd);
      }
    });
  }).wait();

  q.memcpy(output, output_dev, kSize * stride * sizeof(uint16_t)).wait();

  ::sycl::free(input_dev, q);
  ::sycl::free(temp_dev, q);
  ::sycl::free(output_dev, q);
}

// ============================================================================
// Placeholder implementations for larger transforms
// ============================================================================

void fdct16x16(::sycl::queue& q, const int16_t* input, tran_low_t* output,
               int stride, const TxfmParams& params) {
  // TODO: Implement 16x16 DCT using 8-point DCT building blocks
  // For now, fall back to CPU implementation
  constexpr int kSize = 16;
  for (int r = 0; r < kSize; ++r) {
    for (int c = 0; c < kSize; ++c) {
      output[r * kSize + c] = static_cast<tran_low_t>(input[r * stride + c]);
    }
  }
}

void fdct32x32(::sycl::queue& q, const int16_t* input, tran_low_t* output,
               int stride, const TxfmParams& params) {
  // TODO: Implement 32x32 DCT
  constexpr int kSize = 32;
  for (int r = 0; r < kSize; ++r) {
    for (int c = 0; c < kSize; ++c) {
      output[r * kSize + c] = static_cast<tran_low_t>(input[r * stride + c]);
    }
  }
}

void fdct64x64(::sycl::queue& q, const int16_t* input, tran_low_t* output,
               int stride, const TxfmParams& params) {
  // TODO: Implement 64x64 DCT
  constexpr int kSize = 64;
  for (int r = 0; r < kSize; ++r) {
    for (int c = 0; c < kSize; ++c) {
      output[r * kSize + c] = static_cast<tran_low_t>(input[r * stride + c]);
    }
  }
}

void idct16x16(::sycl::queue& q, const tran_low_t* input, uint16_t* output,
               int stride, const TxfmParams& params) {
  // TODO: Implement 16x16 IDCT
  constexpr int kSize = 16;
  const int bd = params.bd;
  for (int r = 0; r < kSize; ++r) {
    for (int c = 0; c < kSize; ++c) {
      output[r * stride + c] = clamp_pixel_kernel(input[r * kSize + c], bd);
    }
  }
}

void idct32x32(::sycl::queue& q, const tran_low_t* input, uint16_t* output,
               int stride, const TxfmParams& params) {
  // TODO: Implement 32x32 IDCT
  constexpr int kSize = 32;
  const int bd = params.bd;
  for (int r = 0; r < kSize; ++r) {
    for (int c = 0; c < kSize; ++c) {
      output[r * stride + c] = clamp_pixel_kernel(input[r * kSize + c], bd);
    }
  }
}

void idct64x64(::sycl::queue& q, const tran_low_t* input, uint16_t* output,
               int stride, const TxfmParams& params) {
  // TODO: Implement 64x64 IDCT
  constexpr int kSize = 64;
  const int bd = params.bd;
  for (int r = 0; r < kSize; ++r) {
    for (int c = 0; c < kSize; ++c) {
      output[r * stride + c] = clamp_pixel_kernel(input[r * kSize + c], bd);
    }
  }
}

// ============================================================================
// ADST Implementations (Placeholders)
// ============================================================================

void fadst4x4(::sycl::queue& q, const int16_t* input, tran_low_t* output,
              int stride, const TxfmParams& params) {
  // TODO: Implement 4x4 ADST
  constexpr int kSize = 4;
  for (int r = 0; r < kSize; ++r) {
    for (int c = 0; c < kSize; ++c) {
      output[r * kSize + c] = static_cast<tran_low_t>(input[r * stride + c]);
    }
  }
}

void fadst8x8(::sycl::queue& q, const int16_t* input, tran_low_t* output,
              int stride, const TxfmParams& params) {
  // TODO: Implement 8x8 ADST
  constexpr int kSize = 8;
  for (int r = 0; r < kSize; ++r) {
    for (int c = 0; c < kSize; ++c) {
      output[r * kSize + c] = static_cast<tran_low_t>(input[r * stride + c]);
    }
  }
}

void fadst16x16(::sycl::queue& q, const int16_t* input, tran_low_t* output,
                int stride, const TxfmParams& params) {
  // TODO: Implement 16x16 ADST
  constexpr int kSize = 16;
  for (int r = 0; r < kSize; ++r) {
    for (int c = 0; c < kSize; ++c) {
      output[r * kSize + c] = static_cast<tran_low_t>(input[r * stride + c]);
    }
  }
}

void iadst4x4(::sycl::queue& q, const tran_low_t* input, uint16_t* output,
              int stride, const TxfmParams& params) {
  // TODO: Implement 4x4 IADST
  constexpr int kSize = 4;
  const int bd = params.bd;
  for (int r = 0; r < kSize; ++r) {
    for (int c = 0; c < kSize; ++c) {
      output[r * stride + c] = clamp_pixel_kernel(input[r * kSize + c], bd);
    }
  }
}

void iadst8x8(::sycl::queue& q, const tran_low_t* input, uint16_t* output,
              int stride, const TxfmParams& params) {
  // TODO: Implement 8x8 IADST
  constexpr int kSize = 8;
  const int bd = params.bd;
  for (int r = 0; r < kSize; ++r) {
    for (int c = 0; c < kSize; ++c) {
      output[r * stride + c] = clamp_pixel_kernel(input[r * kSize + c], bd);
    }
  }
}

void iadst16x16(::sycl::queue& q, const tran_low_t* input, uint16_t* output,
                int stride, const TxfmParams& params) {
  // TODO: Implement 16x16 IADST
  constexpr int kSize = 16;
  const int bd = params.bd;
  for (int r = 0; r < kSize; ++r) {
    for (int c = 0; c < kSize; ++c) {
      output[r * stride + c] = clamp_pixel_kernel(input[r * kSize + c], bd);
    }
  }
}

// ============================================================================
// Identity Transform (IDTX) Implementations
// ============================================================================

void fidtx4x4(::sycl::queue& q, const int16_t* input, tran_low_t* output,
              int stride, const TxfmParams& params) {
  // Identity transform: pass-through with scaling
  constexpr int kSize = 4;
  for (int r = 0; r < kSize; ++r) {
    for (int c = 0; c < kSize; ++c) {
      output[r * kSize + c] = static_cast<tran_low_t>(input[r * stride + c]) * 2;
    }
  }
}

void fidtx8x8(::sycl::queue& q, const int16_t* input, tran_low_t* output,
              int stride, const TxfmParams& params) {
  constexpr int kSize = 8;
  for (int r = 0; r < kSize; ++r) {
    for (int c = 0; c < kSize; ++c) {
      output[r * kSize + c] = static_cast<tran_low_t>(input[r * stride + c]) * 2;
    }
  }
}

void fidtx16x16(::sycl::queue& q, const int16_t* input, tran_low_t* output,
                int stride, const TxfmParams& params) {
  constexpr int kSize = 16;
  for (int r = 0; r < kSize; ++r) {
    for (int c = 0; c < kSize; ++c) {
      output[r * kSize + c] = static_cast<tran_low_t>(input[r * stride + c]) * 2;
    }
  }
}

// ============================================================================
// Hybrid Transform Implementations
// ============================================================================

void hybrid_fwd_txfm(::sycl::queue& q, const int16_t* input, tran_low_t* output,
                     int stride, const TxfmParams& params, TxfmType row_type,
                     TxfmType col_type) {
  // Select appropriate transform based on types
  // For now, default to DCT
  switch (params.tx_size) {
    case TX_4X4:
      fdct4x4(q, input, output, stride, params);
      break;
    case TX_8X8:
      fdct8x8(q, input, output, stride, params);
      break;
    case TX_16X16:
      fdct16x16(q, input, output, stride, params);
      break;
    case TX_32X32:
      fdct32x32(q, input, output, stride, params);
      break;
    case TX_64X64:
      fdct64x64(q, input, output, stride, params);
      break;
    default:
      fdct8x8(q, input, output, stride, params);
      break;
  }
}

void hybrid_inv_txfm(::sycl::queue& q, const tran_low_t* input, uint16_t* output,
                     int stride, const TxfmParams& params, TxfmType row_type,
                     TxfmType col_type) {
  switch (params.tx_size) {
    case TX_4X4:
      idct4x4(q, input, output, stride, params);
      break;
    case TX_8X8:
      idct8x8(q, input, output, stride, params);
      break;
    case TX_16X16:
      idct16x16(q, input, output, stride, params);
      break;
    case TX_32X32:
      idct32x32(q, input, output, stride, params);
      break;
    case TX_64X64:
      idct64x64(q, input, output, stride, params);
      break;
    default:
      idct8x8(q, input, output, stride, params);
      break;
  }
}

// ============================================================================
// Utility Function Implementations
// ============================================================================

int get_tx_scale(TX_SIZE tx_size) {
  switch (tx_size) {
    case TX_4X4:   return 0;
    case TX_8X8:   return 1;
    case TX_16X16: return 2;
    case TX_32X32: return 3;
    case TX_64X64: return 3;
    default:       return 0;
  }
}

}  // namespace sycl
}  // namespace avm

#endif  // HAVE_SYCL
