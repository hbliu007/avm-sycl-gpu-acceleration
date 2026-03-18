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

#include "sycl_lpf.hpp"
#include <algorithm>

namespace avm {
namespace sycl {

// ============================================================================
// Helper Kernels for SYCL
// ============================================================================

namespace {

// Filter decision kernel for determining filter width
class FilterChoiceKernel {
public:
  FilterChoiceKernel(uint16_t* s_ptr, int pitch_val, int max_neg_val,
                     int max_pos_val, uint16_t q_thresh_val,
                     uint16_t side_thresh_val, uint16_t* t_ptr, int* result_ptr)
      : s(s_ptr), pitch(pitch_val), max_neg(max_neg_val),
        max_pos(max_pos_val), q_thresh(q_thresh_val),
        side_thresh(side_thresh_val), t(t_ptr), result(result_ptr) {}

  void operator()(::sycl::item<1> item) const {
    // This kernel determines how many pixels to modify on each side
    // Based on second derivative analysis similar to filt_choice_highbd

    if (!q_thresh || !side_thresh) {
      *result = 0;
      return;
    }

    int max_samples_neg = max_neg == 0 ? 0 : max_neg / 2 - 1;
    int max_samples_pos = max_pos / 2 - 1;

    if (max_samples_pos < 1 || max_samples_pos < max_samples_neg) {
      *result = 0;
      return;
    }

    // Second derivative buffer (using local memory for efficiency)
    int16_t second_deriv_buf[SEC_DERIV_ARRAY_LEN];
    int16_t* second_deriv = &second_deriv_buf[SEC_DERIV_ARRAY_LEN >> 1];

    int8_t mask = 0;

    // Testing for 1 sample modification
    second_deriv[-2] = abs(s[-3 * pitch] - (s[-2 * pitch] << 1) + s[-pitch]);
    second_deriv[1] = abs(s[0] - (s[pitch] << 1) + s[2 * pitch]);

    second_deriv[-2] += abs(t[-3 * pitch] - (t[-2 * pitch] << 1) + t[-pitch]);
    second_deriv[-2] = (second_deriv[-2] + 1) >> 1;

    second_deriv[1] += abs(t[0] - (t[pitch] << 1) + t[2 * pitch]);
    second_deriv[1] = (second_deriv[1] + 1) >> 1;

    mask |= (second_deriv[-2] > side_thresh) * -1;
    mask |= (second_deriv[1] > side_thresh) * -1;

    if (mask) {
      *result = 0;
      return;
    }

    if (max_samples_pos == 1) {
      *result = 1;
      return;
    }

    // Testing for 2 sample modification
    const int side_thresh2 = side_thresh >> 2;

    mask |= (second_deriv[-2] > side_thresh2) * -1;
    mask |= (second_deriv[1] > side_thresh2) * -1;

    second_deriv[-1] = abs(s[-2 * pitch] - (s[-pitch] << 1) + s[0]);
    second_deriv[-1] += abs(t[-2 * pitch] - (t[-pitch] << 1) + t[0]);
    second_deriv[-1] = (second_deriv[-1] + 1) >> 1;

    second_deriv[0] = abs(s[-pitch] - (s[0] << 1) + s[pitch]);
    second_deriv[0] += abs(t[-pitch] - (t[0] << 1) + t[pitch]);
    second_deriv[0] = (second_deriv[0] + 1) >> 1;

    mask |= ((second_deriv[-1] + second_deriv[0]) > q_thresh * DF_6_THRESH) * -1;

    if (mask) {
      *result = 1;
      return;
    }

    // Testing 3 sample modification
    const int side_thresh3 = side_thresh >> FILT_8_THRESH_SHIFT;

    mask |= (second_deriv[-2] > side_thresh3) * -1;
    mask |= (second_deriv[1] > side_thresh3) * -1;

    mask |= ((second_deriv[-1] + second_deriv[0]) > q_thresh * DF_8_THRESH) * -1;

    int end_dir_thresh = (side_thresh * 3) >> 4;

    if (max_samples_neg > 2) {
      int deriv_neg = abs((s[-pitch] - s[-4 * pitch]) - 3 * (s[-pitch] - s[-2 * pitch]));
      int deriv_t_neg = abs((t[-pitch] - t[-4 * pitch]) - 3 * (t[-pitch] - t[-2 * pitch]));
      mask |= (((deriv_neg + deriv_t_neg + 1) >> 1) > end_dir_thresh) * -1;
    }

    int deriv_pos = abs((s[0] - s[3 * pitch]) - 3 * (s[0] - s[pitch]));
    int deriv_t_pos = abs((t[0] - t[3 * pitch]) - 3 * (t[0] - t[pitch]));
    mask |= (((deriv_pos + deriv_t_pos + 1) >> 1) > end_dir_thresh) * -1;

    if (mask) {
      *result = 2;
      return;
    }

    if (max_samples_pos == 3) {
      *result = 3;
      return;
    }

    // Testing 4 sample modification and above
    int transition = (second_deriv[-1] + second_deriv[0]) << DF_Q_THRESH_SHIFT;

    for (int dist = 4; dist < MAX_DBL_FLT_LEN + 1; dist += 2) {
      const int q_thresh4 = q_thresh * q_first[dist - 4];

      mask |= (transition > q_thresh4) * -1;

      end_dir_thresh = (side_thresh * dist) >> 4;

      if (dist == 8) dist = 7;

      if (max_samples_neg >= dist) {
        int deriv_neg = abs((s[-pitch] - s[-(dist + 1) * pitch]) -
                           dist * (s[-pitch] - s[-2 * pitch]));
        int deriv_t_neg = abs((t[-pitch] - t[-(dist + 1) * pitch]) -
                            dist * (t[-pitch] - t[-2 * pitch]));
        mask |= (((deriv_neg + deriv_t_neg + 1) >> 1) > end_dir_thresh) * -1;
      }

      int deriv_pos = abs((s[0] - s[dist * pitch]) - dist * (s[0] - s[pitch]));
      int deriv_t_pos = abs((t[0] - t[dist * pitch]) - dist * (t[0] - t[pitch]));
      mask |= (((deriv_pos + deriv_t_pos + 1) >> 1) > end_dir_thresh) * -1;

      if (dist == 7) dist = 8;

      if (mask) {
        *result = (dist == 4) ? dist - 1 : dist - 2;
        return;
      }
      if (max_samples_pos <= dist) {
        *result = ((dist >> 1) << 1);
        return;
      }
    }

    *result = MAX_DBL_FLT_LEN;
  }

private:
  uint16_t* s;
  int pitch;
  int max_neg;
  int max_pos;
  uint16_t q_thresh;
  uint16_t side_thresh;
  uint16_t* t;
  int* result;
};

// Asymmetric filter kernel
class FilterAsymKernel {
public:
  FilterAsymKernel(uint16_t* s_ptr, int pitch_val, int q_threshold_val,
                   int width_neg_val, int width_pos_val, int bd_val,
                   int is_lossless_neg_val, int is_lossless_pos_val)
      : s(s_ptr), pitch(pitch_val), q_threshold(q_threshold_val),
        width_neg(width_neg_val), width_pos(width_pos_val), bd(bd_val),
        is_lossless_neg(is_lossless_neg_val), is_lossless_pos(is_lossless_pos_val) {}

  void operator()(::sycl::item<1> item) const {
    if (width_neg < 1 || width_pos < 1) return;

    int width = std::max(width_neg, width_pos);
    int delta_m2 = (3 * (s[0] - s[-pitch]) - (s[pitch] - s[-2 * pitch])) * 4;

    int q_thresh_clamp = q_threshold * q_thresh_mults[width - 1];
    delta_m2 = std::min(std::max(delta_m2, -q_thresh_clamp), q_thresh_clamp);

    if (!is_lossless_neg) {
      int delta_m2_neg = delta_m2 * w_mult[width_neg - 1];
      for (int i = 0; i < width_neg; i++) {
        int adjustment = ROUND_POWER_OF_TWO(
            delta_m2_neg * (width_neg - i), 3 + DF_SHIFT);
        s[(-i - 1) * pitch] = clamp_pixel(s[(-i - 1) * pitch] + adjustment, bd);
      }
    }

    if (!is_lossless_pos) {
      int delta_m2_pos = delta_m2 * w_mult[width_pos - 1];
      for (int i = 0; i < width_pos; i++) {
        int adjustment = ROUND_POWER_OF_TWO(
            delta_m2_pos * (width_pos - i), 3 + DF_SHIFT);
        s[i * pitch] = clamp_pixel(s[i * pitch] - adjustment, bd);
      }
    }
  }

private:
  uint16_t* s;
  int pitch;
  int q_threshold;
  int width_neg;
  int width_pos;
  int bd;
  int is_lossless_neg;
  int is_lossless_pos;
};

// Horizontal filter kernel for multiple pixels
class HorizontalFilterKernel {
public:
  HorizontalFilterKernel(uint16_t* s_ptr, int pitch_val, int filter_val,
                         int filt_neg_val, int q_threshold_val, int bd_val,
                         int is_lossless_neg_val, int is_lossless_pos_val,
                         int count_val)
      : s(s_ptr), pitch(pitch_val), filter(filter_val),
        filt_neg(filt_neg_val), q_threshold(q_threshold_val), bd(bd_val),
        is_lossless_neg(is_lossless_neg_val), is_lossless_pos(is_lossless_pos_val),
        count(count_val) {}

  void operator()(::sycl::item<1> item) const {
    int i = item.get_id(0);
    if (i >= count) return;

    uint16_t* s_row = s + i;
    int actual_filter = std::min(filter, filt_neg);

    if (actual_filter < 1) return;

    int width = std::max(1, actual_filter);
    int delta_m2 = (3 * (s_row[0] - s_row[-pitch]) -
                    (s_row[pitch] - s_row[-2 * pitch])) * 4;

    int q_thresh_clamp = q_threshold * q_thresh_mults[width - 1];
    delta_m2 = std::min(std::max(delta_m2, -q_thresh_clamp), q_thresh_clamp);

    if (!is_lossless_neg) {
      int delta_m2_neg = delta_m2 * w_mult[actual_filter - 1];
      for (int j = 0; j < actual_filter; j++) {
        int adjustment = ROUND_POWER_OF_TWO(
            delta_m2_neg * (actual_filter - j), 3 + DF_SHIFT);
        s_row[(-j - 1) * pitch] = clamp_pixel(
            s_row[(-j - 1) * pitch] + adjustment, bd);
      }
    }

    if (!is_lossless_pos) {
      int delta_m2_pos = delta_m2 * w_mult[actual_filter - 1];
      for (int j = 0; j < actual_filter; j++) {
        int adjustment = ROUND_POWER_OF_TWO(
            delta_m2_pos * (actual_filter - j), 3 + DF_SHIFT);
        s_row[j * pitch] = clamp_pixel(s_row[j * pitch] - adjustment, bd);
      }
    }
  }

private:
  uint16_t* s;
  int pitch;
  int filter;
  int filt_neg;
  int q_threshold;
  int bd;
  int is_lossless_neg;
  int is_lossless_pos;
  int count;
};

// Vertical filter kernel for multiple pixels
class VerticalFilterKernel {
public:
  VerticalFilterKernel(uint16_t* s_ptr, int pitch_val, int filter_val,
                       int filt_neg_val, int q_threshold_val, int bd_val,
                       int is_lossless_neg_val, int is_lossless_pos_val,
                       int count_val)
      : s(s_ptr), pitch(pitch_val), filter(filter_val),
        filt_neg(filt_neg_val), q_threshold(q_threshold_val), bd(bd_val),
        is_lossless_neg(is_lossless_neg_val), is_lossless_pos(is_lossless_pos_val),
        count(count_val) {}

  void operator()(::sycl::item<1> item) const {
    int i = item.get_id(0);
    if (i >= count) return;

    uint16_t* s_col = s + i * pitch;
    int actual_filter = std::min(filter, filt_neg);

    if (actual_filter < 1) return;

    int width = std::max(1, actual_filter);
    int delta_m2 = (3 * (s_col[0] - s_col[-1]) - (s_col[1] - s_col[-2])) * 4;

    int q_thresh_clamp = q_threshold * q_thresh_mults[width - 1];
    delta_m2 = std::min(std::max(delta_m2, -q_thresh_clamp), q_thresh_clamp);

    if (!is_lossless_neg) {
      int delta_m2_neg = delta_m2 * w_mult[actual_filter - 1];
      for (int j = 0; j < actual_filter; j++) {
        int adjustment = ROUND_POWER_OF_TWO(
            delta_m2_neg * (actual_filter - j), 3 + DF_SHIFT);
        s_col[-j - 1] = clamp_pixel(s_col[-j - 1] + adjustment, bd);
      }
    }

    if (!is_lossless_pos) {
      int delta_m2_pos = delta_m2 * w_mult[actual_filter - 1];
      for (int j = 0; j < actual_filter; j++) {
        int adjustment = ROUND_POWER_OF_TWO(
            delta_m2_pos * (actual_filter - j), 3 + DF_SHIFT);
        s_col[j] = clamp_pixel(s_col[j] - adjustment, bd);
      }
    }
  }

private:
  uint16_t* s;
  int pitch;
  int filter;
  int filt_neg;
  int q_threshold;
  int bd;
  int is_lossless_neg;
  int is_lossless_pos;
  int count;
};

}  // anonymous namespace

// ============================================================================
// Horizontal Loop Filter Implementations
// ============================================================================

void lpf_horizontal_4(::sycl::queue& q, uint16_t* s, int pitch,
                      const LpfParams& params, int count) {
  int filt_neg = (params.filt_width_neg >> 1) - 1;

  // Allocate device memory and copy if necessary
  uint16_t* s_dev = s;  // Assuming USM or managed memory

  // Temporary buffer for filter decision result
  int* filter_result = sycl::malloc_shared<int>(1, q);

  // Submit filter choice kernel
  q.submit([&](::sycl::handler& cgh) {
    cgh.single_task(FilterChoiceKernel(
        s_dev, pitch, params.filt_width_neg, params.filt_width_pos,
        params.q_thresh, params.side_thresh,
        s_dev + count - 1, filter_result));
  }).wait();

  int filter = *filter_result;
  ::sycl::free(filter_result, q);

  // Apply filter to each pixel in the edge
  q.submit([&](::sycl::handler& cgh) {
    cgh.parallel_for(::sycl::range<1>(count),
        HorizontalFilterKernel(
            s_dev, pitch, filter, filt_neg, params.q_thresh,
            params.bd, params.is_lossless_neg, params.is_lossless_pos, count));
  }).wait();
}

void lpf_horizontal_8(::sycl::queue& q, uint16_t* s, int pitch,
                      const LpfParams& params, int count) {
  int filt_neg = (params.filt_width_neg >> 1) - 1;

  uint16_t* s_dev = s;
  int* filter_result = sycl::malloc_shared<int>(1, q);

  q.submit([&](::sycl::handler& cgh) {
    cgh.single_task(FilterChoiceKernel(
        s_dev, pitch, params.filt_width_neg, params.filt_width_pos,
        params.q_thresh, params.side_thresh,
        s_dev + count - 1, filter_result));
  }).wait();

  int filter = *filter_result;
  ::sycl::free(filter_result, q);

  q.submit([&](::sycl::handler& cgh) {
    cgh.parallel_for(::sycl::range<1>(count),
        HorizontalFilterKernel(
            s_dev, pitch, filter, filt_neg, params.q_thresh,
            params.bd, params.is_lossless_neg, params.is_lossless_pos, count));
  }).wait();
}

void lpf_horizontal_14(::sycl::queue& q, uint16_t* s, int pitch,
                       const LpfParams& params, int count) {
  int filt_neg = (params.filt_width_neg >> 1) - 1;

  uint16_t* s_dev = s;
  int* filter_result = sycl::malloc_shared<int>(1, q);

  q.submit([&](::sycl::handler& cgh) {
    cgh.single_task(FilterChoiceKernel(
        s_dev, pitch, params.filt_width_neg, params.filt_width_pos,
        params.q_thresh, params.side_thresh,
        s_dev + count - 1, filter_result));
  }).wait();

  int filter = *filter_result;
  ::sycl::free(filter_result, q);

  q.submit([&](::sycl::handler& cgh) {
    cgh.parallel_for(::sycl::range<1>(count),
        HorizontalFilterKernel(
            s_dev, pitch, filter, filt_neg, params.q_thresh,
            params.bd, params.is_lossless_neg, params.is_lossless_pos, count));
  }).wait();
}

// ============================================================================
// Vertical Loop Filter Implementations
// ============================================================================

void lpf_vertical_4(::sycl::queue& q, uint16_t* s, int pitch,
                    const LpfParams& params, int count) {
  int filt_neg = (params.filt_width_neg >> 1) - 1;

  uint16_t* s_dev = s;
  int* filter_result = sycl::malloc_shared<int>(1, q);

  // For vertical filter, pitch = 1 (column stride)
  q.submit([&](::sycl::handler& cgh) {
    cgh.single_task(FilterChoiceKernel(
        s_dev, 1, params.filt_width_neg, params.filt_width_pos,
        params.q_thresh, params.side_thresh,
        s_dev + (count - 1) * pitch, filter_result));
  }).wait();

  int filter = *filter_result;
  ::sycl::free(filter_result, q);

  q.submit([&](::sycl::handler& cgh) {
    cgh.parallel_for(::sycl::range<1>(count),
        VerticalFilterKernel(
            s_dev, pitch, filter, filt_neg, params.q_thresh,
            params.bd, params.is_lossless_neg, params.is_lossless_pos, count));
  }).wait();
}

void lpf_vertical_8(::sycl::queue& q, uint16_t* s, int pitch,
                    const LpfParams& params, int count) {
  int filt_neg = (params.filt_width_neg >> 1) - 1;

  uint16_t* s_dev = s;
  int* filter_result = sycl::malloc_shared<int>(1, q);

  q.submit([&](::sycl::handler& cgh) {
    cgh.single_task(FilterChoiceKernel(
        s_dev, 1, params.filt_width_neg, params.filt_width_pos,
        params.q_thresh, params.side_thresh,
        s_dev + (count - 1) * pitch, filter_result));
  }).wait();

  int filter = *filter_result;
  ::sycl::free(filter_result, q);

  q.submit([&](::sycl::handler& cgh) {
    cgh.parallel_for(::sycl::range<1>(count),
        VerticalFilterKernel(
            s_dev, pitch, filter, filt_neg, params.q_thresh,
            params.bd, params.is_lossless_neg, params.is_lossless_pos, count));
  }).wait();
}

void lpf_vertical_14(::sycl::queue& q, uint16_t* s, int pitch,
                     const LpfParams& params, int count) {
  int filt_neg = (params.filt_width_neg >> 1) - 1;

  uint16_t* s_dev = s;
  int* filter_result = sycl::malloc_shared<int>(1, q);

  q.submit([&](::sycl::handler& cgh) {
    cgh.single_task(FilterChoiceKernel(
        s_dev, 1, params.filt_width_neg, params.filt_width_pos,
        params.q_thresh, params.side_thresh,
        s_dev + (count - 1) * pitch, filter_result));
  }).wait();

  int filter = *filter_result;
  ::sycl::free(filter_result, q);

  q.submit([&](::sycl::handler& cgh) {
    cgh.parallel_for(::sycl::range<1>(count),
        VerticalFilterKernel(
            s_dev, pitch, filter, filt_neg, params.q_thresh,
            params.bd, params.is_lossless_neg, params.is_lossless_pos, count));
  }).wait();
}

// ============================================================================
// Dual Filter Variants
// ============================================================================

void lpf_dual(::sycl::queue& q, uint16_t* s, int pitch,
              const LpfParams& params_h, const LpfParams& params_v) {
  // Apply horizontal filter first
  lpf_horizontal_4(q, s, pitch, params_h, 4);

  // Then apply vertical filter
  lpf_vertical_4(q, s, pitch, params_v, 4);
}

void lpf_horizontal_edge_8(::sycl::queue& q, uint16_t* s, int pitch,
                           const LpfParams& params, int width, int height) {
  for (int row = 0; row < height; row++) {
    uint16_t* edge = s + row * pitch;
    lpf_horizontal_8(q, edge, pitch, params, width);
  }
}

void lpf_vertical_edge_8(::sycl::queue& q, uint16_t* s, int pitch,
                         const LpfParams& params, int width, int height) {
  for (int col = 0; col < width; col++) {
    uint16_t* edge = s + col;
    lpf_vertical_8(q, edge, pitch, params, height);
  }
}

// ============================================================================
// Filter Selection Function
// ============================================================================

int filt_choice(::sycl::queue& q, uint16_t* s, int pitch,
                int max_filt_neg, int max_filt_pos,
                uint16_t q_thresh, uint16_t side_thresh, uint16_t* t) {
  int* result = sycl::malloc_shared<int>(1, q);

  q.submit([&](::sycl::handler& cgh) {
    cgh.single_task(FilterChoiceKernel(
        s, pitch, max_filt_neg, max_filt_pos,
        q_thresh, side_thresh, t, result));
  }).wait();

  int filter_width = *result;
  ::sycl::free(result, q);

  return filter_width;
}

// ============================================================================
// Batch Processing Functions
// ============================================================================

void batch_lpf_horizontal(::sycl::queue& q, uint16_t** edges, int pitch,
                          const LpfParams* params, int num_edges, int width) {
  for (int i = 0; i < num_edges; i++) {
    switch (width) {
      case 4:
        lpf_horizontal_4(q, edges[i], pitch, params[i], 4);
        break;
      case 8:
        lpf_horizontal_8(q, edges[i], pitch, params[i], 8);
        break;
      case 14:
        lpf_horizontal_14(q, edges[i], pitch, params[i], 4);
        break;
      default:
        break;
    }
  }
}

void batch_lpf_vertical(::sycl::queue& q, uint16_t** edges, int pitch,
                        const LpfParams* params, int num_edges, int width) {
  for (int i = 0; i < num_edges; i++) {
    switch (width) {
      case 4:
        lpf_vertical_4(q, edges[i], pitch, params[i], 4);
        break;
      case 8:
        lpf_vertical_8(q, edges[i], pitch, params[i], 8);
        break;
      case 14:
        lpf_vertical_14(q, edges[i], pitch, params[i], 4);
        break;
      default:
        break;
    }
  }
}

// ============================================================================
// Utility Function Implementations
// ============================================================================

void apply_filter_asym(uint16_t* s, int pitch, int delta_m2,
                       int width, int q_thresh_clamp, int bd,
                       int is_lossless_neg, int is_lossless_pos) {
  delta_m2 = std::min(std::max(delta_m2, -q_thresh_clamp), q_thresh_clamp);

  if (!is_lossless_neg) {
    int delta_m2_neg = delta_m2 * w_mult[width - 1];
    for (int i = 0; i < width; i++) {
      int adjustment = ROUND_POWER_OF_TWO(
          delta_m2_neg * (width - i), 3 + DF_SHIFT);
      s[(-i - 1) * pitch] = clamp_pixel(s[(-i - 1) * pitch] + adjustment, bd);
    }
  }

  if (!is_lossless_pos) {
    int delta_m2_pos = delta_m2 * w_mult[width - 1];
    for (int i = 0; i < width; i++) {
      int adjustment = ROUND_POWER_OF_TWO(
          delta_m2_pos * (width - i), 3 + DF_SHIFT);
      s[i * pitch] = clamp_pixel(s[i * pitch] - adjustment, bd);
    }
  }
}

}  // namespace sycl
}  // namespace avm
