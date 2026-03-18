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

#include "sycl_intra.hpp"

#ifdef HAVE_SYCL

#include <sycl/sycl.hpp>
#include "sycl_context.hpp"

namespace avm {
namespace sycl {

namespace {

// DC prediction kernel - broadcasts a single DC value to all pixels
template <int BLOCK_SIZE>
class DcPredKernel {
public:
  DcPredKernel(uint8_t dc_val, uint8_t* dst_ptr, int stride_val)
      : dc(dc_val), dst(dst_ptr), stride(stride_val) {}

  void operator()(::sycl::item<2> item) const {
    int row = item.get_id(0);
    int col = item.get_id(1);
    dst[row * stride + col] = dc;
  }

private:
  uint8_t dc;
  uint8_t* dst;
  int stride;
};

// Horizontal prediction kernel - fills each row with its left reference pixel
template <int BLOCK_SIZE>
class HPredKernel {
public:
  HPredKernel(const uint8_t* ref_ptr, uint8_t* dst_ptr, int stride_val)
      : ref(ref_ptr), dst(dst_ptr), stride(stride_val) {}

  void operator()(::sycl::item<2> item) const {
    int row = item.get_id(0);
    int col = item.get_id(1);
    // ref points to left reference column at row offset
    dst[row * stride + col] = ref[row];
  }

private:
  const uint8_t* ref;
  uint8_t* dst;
  int stride;
};

// Vertical prediction kernel - fills each column with its top reference pixel
template <int BLOCK_SIZE>
class VPredKernel {
public:
  VPredKernel(const uint8_t* ref_ptr, uint8_t* dst_ptr, int stride_val)
      : ref(ref_ptr), dst(dst_ptr), stride(stride_val) {}

  void operator()(::sycl::item<2> item) const {
    int row = item.get_id(0);
    int col = item.get_id(1);
    // ref points to top reference row
    dst[row * stride + col] = ref[col];
  }

private:
  const uint8_t* ref;
  uint8_t* dst;
  int stride;
};

// Helper to compute DC value from reference pixels
template <int BLOCK_SIZE>
uint8_t compute_dc_value(::sycl::queue& q, const uint8_t* ref) {
  // For simplicity, we compute DC on host using the combined above and left refs
  // ref layout: [above pixels (BLOCK_SIZE)] + [left pixels (BLOCK_SIZE)]
  int sum = 0;
  for (int i = 0; i < 2 * BLOCK_SIZE; ++i) {
    sum += ref[i];
  }
  return static_cast<uint8_t>((sum + BLOCK_SIZE) / (2 * BLOCK_SIZE));
}

}  // anonymous namespace

// ============================================================================
// DC Prediction Implementations (8x8 implemented with SYCL)
// ============================================================================

void intra_pred_dc_4x4(const uint8_t* ref, uint8_t* dst, int stride) {
  ::sycl::queue& q = SYCLContext::instance().queue();

  // Compute DC value on host
  int sum = 0;
  for (int i = 0; i < 8; ++i) {  // 4 above + 4 left
    sum += ref[i];
  }
  uint8_t dc = static_cast<uint8_t>((sum + 2) / 8);

  // Broadcast DC value to all pixels using single_task
  q.submit([&](::sycl::handler& cgh) {
    cgh.single_task([=]() {
      for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
          dst[row * stride + col] = dc;
        }
      }
    });
  }).wait();
}

void intra_pred_dc_8x8(const uint8_t* ref, uint8_t* dst, int stride) {
  ::sycl::queue& q = SYCLContext::instance().queue();

  // Compute DC value on host
  int sum = 0;
  for (int i = 0; i < 16; ++i) {  // 8 above + 8 left
    sum += ref[i];
  }
  uint8_t dc = static_cast<uint8_t>((sum + 4) / 16);

  // Broadcast DC value using single_task
  q.submit([&](::sycl::handler& cgh) {
    cgh.single_task([=]() {
      for (int row = 0; row < 8; ++row) {
        for (int col = 0; col < 8; ++col) {
          dst[row * stride + col] = dc;
        }
      }
    });
  }).wait();
}

void intra_pred_dc_16x16(const uint8_t* ref, uint8_t* dst, int stride) {
  ::sycl::queue& q = SYCLContext::instance().queue();

  // Compute DC value on host
  int sum = 0;
  for (int i = 0; i < 32; ++i) {  // 16 above + 16 left
    sum += ref[i];
  }
  uint8_t dc = static_cast<uint8_t>((sum + 8) / 32);

  // Broadcast DC value using single_task
  q.submit([&](::sycl::handler& cgh) {
    cgh.single_task([=]() {
      for (int row = 0; row < 16; ++row) {
        for (int col = 0; col < 16; ++col) {
          dst[row * stride + col] = dc;
        }
      }
    });
  }).wait();
}

void intra_pred_dc_32x32(const uint8_t* ref, uint8_t* dst, int stride) {
  ::sycl::queue& q = SYCLContext::instance().queue();

  // Compute DC value on host
  int sum = 0;
  for (int i = 0; i < 64; ++i) {  // 32 above + 32 left
    sum += ref[i];
  }
  uint8_t dc = static_cast<uint8_t>((sum + 16) / 64);

  // Broadcast DC value using single_task
  q.submit([&](::sycl::handler& cgh) {
    cgh.single_task([=]() {
      for (int row = 0; row < 32; ++row) {
        for (int col = 0; col < 32; ++col) {
          dst[row * stride + col] = dc;
        }
      }
    });
  }).wait();
}

// ============================================================================
// Horizontal Prediction Implementations (8x8 implemented with SYCL)
// ============================================================================

void intra_pred_h_4x4(const uint8_t* ref, uint8_t* dst, int stride) {
  ::sycl::queue& q = SYCLContext::instance().queue();

  // ref points to left reference column
  q.submit([&](::sycl::handler& cgh) {
    cgh.parallel_for(::sycl::range<2>(4, 4),
        HPredKernel<4>(ref, dst, stride));
  }).wait();
}

void intra_pred_h_8x8(const uint8_t* ref, uint8_t* dst, int stride) {
  ::sycl::queue& q = SYCLContext::instance().queue();

  // ref points to left reference column
  q.submit([&](::sycl::handler& cgh) {
    cgh.parallel_for(::sycl::range<2>(8, 8),
        HPredKernel<8>(ref, dst, stride));
  }).wait();
}

void intra_pred_h_16x16(const uint8_t* ref, uint8_t* dst, int stride) {
  ::sycl::queue& q = SYCLContext::instance().queue();

  // ref points to left reference column
  q.submit([&](::sycl::handler& cgh) {
    cgh.parallel_for(::sycl::range<2>(16, 16),
        HPredKernel<16>(ref, dst, stride));
  }).wait();
}

void intra_pred_h_32x32(const uint8_t* ref, uint8_t* dst, int stride) {
  ::sycl::queue& q = SYCLContext::instance().queue();

  // ref points to left reference column
  q.submit([&](::sycl::handler& cgh) {
    cgh.parallel_for(::sycl::range<2>(32, 32),
        HPredKernel<32>(ref, dst, stride));
  }).wait();
}

// ============================================================================
// Vertical Prediction Implementations (8x8 implemented with SYCL)
// ============================================================================

void intra_pred_v_4x4(const uint8_t* ref, uint8_t* dst, int stride) {
  ::sycl::queue& q = SYCLContext::instance().queue();

  // ref points to top reference row
  q.submit([&](::sycl::handler& cgh) {
    cgh.parallel_for(::sycl::range<2>(4, 4),
        VPredKernel<4>(ref, dst, stride));
  }).wait();
}

void intra_pred_v_8x8(const uint8_t* ref, uint8_t* dst, int stride) {
  ::sycl::queue& q = SYCLContext::instance().queue();

  // ref points to top reference row
  q.submit([&](::sycl::handler& cgh) {
    cgh.parallel_for(::sycl::range<2>(8, 8),
        VPredKernel<8>(ref, dst, stride));
  }).wait();
}

void intra_pred_v_16x16(const uint8_t* ref, uint8_t* dst, int stride) {
  ::sycl::queue& q = SYCLContext::instance().queue();

  // ref points to top reference row
  q.submit([&](::sycl::handler& cgh) {
    cgh.parallel_for(::sycl::range<2>(16, 16),
        VPredKernel<16>(ref, dst, stride));
  }).wait();
}

void intra_pred_v_32x32(const uint8_t* ref, uint8_t* dst, int stride) {
  ::sycl::queue& q = SYCLContext::instance().queue();

  // ref points to top reference row
  q.submit([&](::sycl::handler& cgh) {
    cgh.parallel_for(::sycl::range<2>(32, 32),
        VPredKernel<32>(ref, dst, stride));
  }).wait();
}

}  // namespace sycl
}  // namespace avm

#endif  // HAVE_SYCL
