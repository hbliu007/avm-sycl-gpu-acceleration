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

#ifndef AVM_AVM_DSP_SYCL_SYCL_TXFM_HPP_
#define AVM_AVM_DSP_SYCL_SYCL_TXFM_HPP_

#include <sycl/sycl.hpp>
#include "av2/common/enums.h"

namespace avm {
namespace sycl {

// Transform types matching AV2 specification
enum class TxfmType {
  kDct2,      // DCT-II
  kAdst,      // Asymmetric Discrete Sine Transform
  kFlipAdst,  // Flipped ADST
  kIdtx,      // Identity Transform
  kWht,       // Walsh-Hadamard Transform
  kFadst4,    // 4-point Fast ADST
  kFadst8     // 8-point Fast ADST
};

// Transform direction
enum class TxfmDir {
  kForward,
  kInverse
};

// Transform parameters for SYCL kernels
struct TxfmParams {
  TX_SIZE tx_size;
  TX_TYPE tx_type;
  int bd;              // Bit depth
  int lossless;
  int eob;             // End of block (for inverse)
  TxfmDir dir;
  int shift;
};

// ============================================================================
// Forward DCT (Discrete Cosine Transform - Type II)
// ============================================================================

/// @brief 2D Forward 4x4 DCT-II
/// @param q Queue for SYCL execution
/// @param input Input residual buffer (src_diff)
/// @param output Output coefficient buffer
/// @param stride Input stride in pixels
/// @param params Transform parameters
void fdct4x4(::sycl::queue& q, const int16_t* input, tran_low_t* output,
             int stride, const TxfmParams& params);

/// @brief 2D Forward 8x8 DCT-II
void fdct8x8(::sycl::queue& q, const int16_t* input, tran_low_t* output,
             int stride, const TxfmParams& params);

/// @brief 2D Forward 16x16 DCT-II
void fdct16x16(sycl:: queue& q, const int16_t* input, tran_low_t* output,
               int stride, const TxfmParams& params);

/// @brief 2D Forward 32x32 DCT-II
void fdct32x32(::sycl::queue& q, const int16_t* input, tran_low_t* output,
               int stride, const TxfmParams& params);

/// @brief 2D Forward 64x64 DCT-II
void fdct64x64(::sycl::queue& q, const int16_t* input, tran_low_t* output,
               int stride, const TxfmParams& params);

// ============================================================================
// Inverse DCT (Discrete Cosine Transform - Type III)
// ============================================================================

/// @brief 2D Inverse 4x4 DCT
/// @param q Queue for SYCL execution
/// @param input Input coefficient buffer (dqcoeff)
/// @param output Output reconstructed buffer
/// @param stride Output stride in pixels
/// @param params Transform parameters
void idct4x4(::sycl::queue& q, const tran_low_t* input, uint16_t* output,
             int stride, const TxfmParams& params);

/// @brief 2D Inverse 8x8 DCT
void idct8x8(::sycl::queue& q, const tran_low_t* input, uint16_t* output,
             int stride, const TxfmParams& params);

/// @brief 2D Inverse 16x16 DCT
void idct16x16(::sycl::queue& q, const tran_low_t* input, uint16_t* output,
               int stride, const TxfmParams& params);

/// @brief 2D Inverse 32x32 DCT
void idct32x32(::sycl::queue& q, const tran_low_t* input, uint16_t* output,
               int stride, const TxfmParams& params);

/// @brief 2D Inverse 64x64 DCT
void idct64x64(::sycl::queue& q, const tran_low_t* input, uint16_t* output,
               int stride, const TxfmParams& params);

// ============================================================================
// Forward ADST (Asymmetric Discrete Sine Transform)
// ============================================================================

/// @brief 2D Forward 4x4 ADST
/// @param q Queue for SYCL execution
/// @param input Input residual buffer
/// @param output Output coefficient buffer
/// @param stride Input stride in pixels
/// @param params Transform parameters
void fadst4x4(::sycl::queue& q, const int16_t* input, tran_low_t* output,
              int stride, const TxfmParams& params);

/// @brief 2D Forward 8x8 ADST
void fadst8x8(::sycl::queue& q, const int16_t* input, tran_low_t* output,
              int stride, const TxfmParams& params);

/// @brief 2D Forward 16x16 ADST
void fadst16x16(::sycl::queue& q, const int16_t* input, tran_low_t* output,
                int stride, const TxfmParams& params);

// ============================================================================
// Inverse ADST
// ============================================================================

/// @brief 2D Inverse 4x4 ADST
/// @param q Queue for SYCL execution
/// @param input Input coefficient buffer
/// @param output Output reconstructed buffer
/// @param stride Output stride in pixels
/// @param params Transform parameters
void iadst4x4(::sycl::queue& q, const tran_low_t* input, uint16_t* output,
              int stride, const TxfmParams& params);

/// @brief 2D Inverse 8x8 ADST
void iadst8x8(::sycl::queue& q, const tran_low_t* input, uint16_t* output,
              int stride, const TxfmParams& params);

/// @brief 2D Inverse 16x16 ADST
void iadst16x16(::sycl::queue& q, const tran_low_t* input, uint16_t* output,
                int stride, const TxfmParams& params);

// ============================================================================
// Identity Transform (IDTX)
// ============================================================================

/// @brief 2D Forward/Inverse 4x4 Identity Transform
/// @param q Queue for SYCL execution
/// @param input Input buffer (residual or coefficients)
/// @param output Output buffer
/// @param stride Stride in pixels
/// @param params Transform parameters
void fidtx4x4(::sycl::queue& q, const int16_t* input, tran_low_t* output,
              int stride, const TxfmParams& params);

/// @brief 2D Forward/Inverse 8x8 Identity Transform
void fidtx8x8(::sycl::queue& q, const int16_t* input, tran_low_t* output,
              int stride, const TxfmParams& params);

/// @brief 2D Forward/Inverse 16x16 Identity Transform
void fidtx16x16(::sycl::queue& q, const int16_t* input, tran_low_t* output,
                int stride, const TxfmParams& params);

// ============================================================================
// Hybrid Transform (for different TX_TYPE combinations)
// ============================================================================

/// @brief 2D Hybrid Forward Transform (row/col types can differ)
/// @param q Queue for SYCL execution
/// @param input Input residual buffer
/// @param output Output coefficient buffer
/// @param stride Input stride in pixels
/// @param params Transform parameters with row_type and col_type
void hybrid_fwd_txfm(::sycl::queue& q, const int16_t* input, tran_low_t* output,
                     int stride, const TxfmParams& params, TxfmType row_type,
                     TxfmType col_type);

/// @brief 2D Hybrid Inverse Transform
/// @param q Queue for SYCL execution
/// @param input Input coefficient buffer
/// @param output Output reconstructed buffer
/// @param stride Output stride in pixels
/// @param params Transform parameters with row_type and col_type
void hybrid_inv_txfm(::sycl::queue& q, const tran_low_t* input, uint16_t* output,
                     int stride, const TxfmParams& params, TxfmType row_type,
                     TxfmType col_type);

// ============================================================================
// Utility Functions
// ============================================================================

/// @brief Get transform scale factor based on size
/// @param tx_size Transform size
/// @return Scale shift value (0-3)
int get_tx_scale(TX_SIZE tx_size);

/// @brief Round and shift for transform coefficients
/// @param value Input value
/// @param shift Right shift amount
/// @return Rounded and shifted value
inline tran_low_t round_shift(tran_high_t value, int shift) {
  assert(shift >= 0);
  return static_cast<tran_low_t>((value + (1 << (shift - 1))) >> shift);
}

/// @brief Clamp value to valid range for given bit depth
/// @param value Input value
/// @param bd Bit depth (8, 10, or 12)
/// @return Clamped value
inline uint16_t clamp_pixel(int value, int bd) {
  const int max = (1 << bd) - 1;
  return static_cast<uint16_t>(sycl::clamp(value, 0, max));
}

}  // namespace sycl
}  // namespace avm

#endif  // AVM_AVM_DSP_SYCL_SYCL_TXFM_HPP_
