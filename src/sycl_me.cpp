/*
 * Copyright (c) 2021, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 3-Clause Clear License
 * and the Alliance for Open Media Patent License 1.0. If the BSD 3-Clause Clear
 * License was not distributed with this source code in the LICENSE file, you
 * can obtain it at aomedia.org/license/software-license/bsd-3-c-c/.  If the
 * Alliance for Open Media Patent License 1.0 was not distributed with this
 * source code in the PATENTS file, you can obtain it at aomedia.org/license/patent-license/.
 */

#include "sycl_me.hpp"
#include <algorithm>
#include <cassert>

namespace avm {
namespace sycl {

// ============================================================================
// Local SAD Kernel - 4x4 with local memory optimization
// ============================================================================

class SAD4x4Kernel;
class SAD16x16Kernel;

/// @brief 4x4 SAD kernel using local memory for cache optimization
/// Each work-group processes one position, work-items process rows in parallel
void sad4x4(sycl::queue& q, const uint16_t* src, int src_stride,
            const uint16_t* ref, int ref_stride, int ref_x, int ref_y,
            uint32_t* results, const MEParams& params) {
  const int block_size = 4;
  const int search_range = params.search_range;
  const int num_positions = (2 * search_range + 1) * (2 * search_range + 1);

  // Local memory for caching source and reference blocks
  constexpr size_t local_src_size = 4 * 4;
  constexpr size_t local_ref_size = 4 * 4;

  sycl::buffer<uint16_t, 1> src_buf(src, src_stride * block_size);
  sycl::buffer<uint16_t, 1> ref_buf(ref, ref_stride * (block_size + 2 * search_range));
  sycl::buffer<uint32_t, 1> results_buf(results, num_positions);

  q.submit([&](sycl::handler& cgh) {
    auto src_acc = src_buf.get_access<sycl::access::mode::read>(cgh);
    auto ref_acc = ref_buf.get_access<sycl::access::mode::read>(cgh);
    auto results_acc = results_buf.get_access<sycl::access::mode::write>(cgh);

    sycl::accessor<uint16_t, 1, sycl::access::mode::read_write, sycl::target::device> local_src(
      sycl::range<1>(local_src_size), cgh);
    sycl::accessor<uint16_t, 1, sycl::access::mode::read_write, sycl::target::device> local_ref(
      sycl::range<1>(local_ref_size), cgh);

    // Each work-group processes one search position
    // Work-items within group process rows in parallel
    auto kern = [&](sycl::nd_item<1> item) {
      const int gid = item.get_global_id(0);
      const int lid = item.get_local_id(0);
      const int group_size = item.get_local_range(0);

      if (gid >= num_positions) return;

      // Compute search position offset
      const int dy = (gid / (2 * search_range + 1)) - search_range;
      const int dx = (gid % (2 * search_range + 1)) - search_range;

      const int ref_offset = (ref_y + dy) * ref_stride + (ref_x + dx);

      // Load source block into local memory (all groups need same source)
      for (int i = lid; i < local_src_size; i += group_size) {
        const int y = i / block_size;
        const int x = i % block_size;
        local_src[i] = src_acc[y * src_stride + x];
      }

      // Load reference block into local memory
      for (int i = lid; i < local_ref_size; i += group_size) {
        const int y = i / block_size;
        const int x = i % block_size;
        local_ref[i] = ref_acc[(ref_y + dy + y) * ref_stride + (ref_x + dx + x)];
      }

      item.barrier(sycl::access::fence_space::local_space);

      // Compute SAD with parallel reduction
      uint32_t partial_sad = 0;
      for (int i = lid; i < local_src_size; i += group_size) {
        partial_sad += sycl::abs(local_src[i] - local_ref[i]);
      }

      // Reduce within work-group
      auto group_sad = sycl::reduce_over_group(item.get_group(), partial_sad, sycl::plus<>());

      if (lid == 0) {
        results_acc[gid] = group_sad;
      }
    };

    const int local_size = 16;  // Optimal for most GPUs
    const int global_size = ((num_positions + local_size - 1) / local_size) * local_size;
    cgh.parallel_for<class SAD4x4Kernel>(
      sycl::nd_range<1>(global_size, local_size), kern);
  });
}

/// @brief 16x16 SAD kernel with optimized local memory access
void sad16x16(sycl::queue& q, const uint16_t* src, int src_stride,
              const uint16_t* ref, int ref_stride, int ref_x, int ref_y,
              uint32_t* results, const MEParams& params) {
  const int block_size = 16;
  const int search_range = params.search_range;
  const int num_positions = (2 * search_range + 1) * (2 * search_range + 1);

  // Use 2D work-groups for 16x16 blocks
  sycl::range<2> local_size{16, 16};  // 16x16 work-group

  sycl::buffer<uint16_t, 2> src_buf((uint16_t*)src, sycl::range<2>{block_size, block_size},
                                     {src_stride, 1});
  sycl::buffer<uint16_t, 2> ref_buf((uint16_t*)ref,
                                     sycl::range<2>{block_size + 2*search_range, block_size + 2*search_range},
                                     {ref_stride, 1});
  sycl::buffer<uint32_t, 1> results_buf(results, num_positions);

  q.submit([&](sycl::handler& cgh) {
    auto src_acc = src_buf.get_access<sycl::access::mode::read>(cgh);
    auto ref_acc = ref_buf.get_access<sycl::access::mode::read>(cgh);
    auto results_acc = results_buf.get_access<sycl::access::mode::write>(cgh);

    // Local memory for tiling
    sycl::accessor<uint16_t, 2, sycl::access::mode::read_write, sycl::target::device> local_src(
      sycl::range<2>{16, 16}, cgh);
    sycl::accessor<uint16_t, 2, sycl::access::mode::read_write, sycl::target::device> local_ref(
      sycl::range<2>{16, 16}, cgh);

    auto kern = [&](sycl::nd_item<2> item) {
      const int gid = item.get_group(0) * item.get_group_range(1) + item.get_group(1);
      const int lid_x = item.get_local_id(0);
      const int lid_y = item.get_local_id(1);

      if (gid >= num_positions) return;

      // Compute search position offset
      const int dy = (gid / (2 * search_range + 1)) - search_range;
      const int dx = (gid % (2 * search_range + 1)) - search_range;

      // Load source tile (coalesced access)
      local_src[lid_y][lid_x] = src_acc[lid_y][lid_x];

      // Load reference tile with offset
      local_ref[lid_y][lid_x] = ref_acc[ref_y + dy + lid_y][ref_x + dx + lid_x];

      item.barrier(sycl::access::fence_space::local_space);

      // Compute SAD for this pixel
      uint32_t pixel_sad = sycl::abs(local_src[lid_y][lid_x] - local_ref[lid_y][lid_x]);

      // Reduce within work-group (2D tree reduction)
      auto group = item.get_group();
      uint32_t tile_sad = sycl::reduce_over_group(group, pixel_sad, sycl::plus<>());

      // Write result (only first work-item)
      if (lid_x == 0 && lid_y == 0) {
        results_acc[gid] = tile_sad;
      }
    };

    const int num_groups_x = (2 * search_range + 1);
    const int num_groups_y = (2 * search_range + 1);
    cgh.parallel_for<class SAD16x16Kernel>(
      sycl::nd_range<2>{num_groups_y * 16, num_groups_x * 16},
      sycl::nd_range<2>{16, 16}, kern);
  });
}

/// @brief 8x8 SAD kernel
void sad8x8(sycl::queue& q, const uint16_t* src, int src_stride,
            const uint16_t* ref, int ref_stride, int ref_x, int ref_y,
            uint32_t* results, const MEParams& params) {
  const int block_size = 8;
  const int search_range = params.search_range;
  const int num_positions = (2 * search_range + 1) * (2 * search_range + 1);

  // 8x8 work-group with 2D local memory
  sycl::range<2> local_size{8, 8};

  sycl::buffer<uint16_t, 2> src_buf((uint16_t*)src, sycl::range<2>{block_size, block_size},
                                     {src_stride, 1});
  sycl::buffer<uint16_t, 2> ref_buf((uint16_t*)ref,
                                     sycl::range<2>{block_size + 2*search_range, block_size + 2*search_range},
                                     {ref_stride, 1});
  sycl::buffer<uint32_t, 1> results_buf(results, num_positions);

  q.submit([&](sycl::handler& cgh) {
    auto src_acc = src_buf.get_access<sycl::access::mode::read>(cgh);
    auto ref_acc = ref_buf.get_access<sycl::access::mode::read>(cgh);
    auto results_acc = results_buf.get_access<sycl::access::mode::write>(cgh);

    sycl::accessor<uint16_t, 2, sycl::access::mode::read_write, sycl::target::device> local_src(
      sycl::range<2>{8, 8}, cgh);
    sycl::accessor<uint16_t, 2, sycl::access::mode::read_write, sycl::target::device> local_ref(
      sycl::range<2>{8, 8}, cgh);

    auto kern = [&](sycl::nd_item<2> item) {
      const int gid = item.get_group(0) * item.get_group_range(1) + item.get_group(1);
      const int lid_x = item.get_local_id(0);
      const int lid_y = item.get_local_id(1);

      if (gid >= num_positions) return;

      const int dy = (gid / (2 * search_range + 1)) - search_range;
      const int dx = (gid % (2 * search_range + 1)) - search_range;

      local_src[lid_y][lid_x] = src_acc[lid_y][lid_x];
      local_ref[lid_y][lid_x] = ref_acc[ref_y + dy + lid_y][ref_x + dx + lid_x];

      item.barrier(sycl::access::fence_space::local_space);

      uint32_t pixel_sad = sycl::abs(local_src[lid_y][lid_x] - local_ref[lid_y][lid_x]);
      auto group = item.get_group();
      uint32_t tile_sad = sycl::reduce_over_group(group, pixel_sad, sycl::plus<>());

      if (lid_x == 0 && lid_y == 0) {
        results_acc[gid] = tile_sad;
      }
    };

    const int num_groups_x = (2 * search_range + 1);
    const int num_groups_y = (2 * search_range + 1);
    cgh.parallel_for<sycl::submit<class SAD8x8Kernel>>(
      sycl::nd_range<2>{num_groups_y * 8, num_groups_x * 8},
      sycl::nd_range<2>{8, 8}, kern);
  });
}

/// @brief 32x32 SAD kernel (using 16x16 tiles)
void sad32x32(sycl::queue& q, const uint16_t* src, int src_stride,
              const uint16_t* ref, int ref_stride, int ref_x, int ref_y,
              uint32_t* results, const MEParams& params) {
  const int block_size = 32;
  const int search_range = params.search_range;
  const int num_positions = (2 * search_range + 1) * (2 * search_range + 1);

  // Process 32x32 as four 16x16 tiles
  sycl::range<2> local_size{16, 16};

  sycl::buffer<uint16_t, 2> src_buf((uint16_t*)src, sycl::range<2>{block_size, block_size},
                                     {src_stride, 1});
  sycl::buffer<uint16_t, 2> ref_buf((uint16_t*)ref,
                                     sycl::range<2>{block_size + 2*search_range, block_size + 2*search_range},
                                     {ref_stride, 1});
  sycl::buffer<uint32_t, 1> results_buf(results, num_positions);

  q.submit([&](sycl::handler& cgh) {
    auto src_acc = src_buf.get_access<sycl::access::mode::read>(cgh);
    auto ref_acc = ref_buf.get_access<sycl::access::mode::read>(cgh);
    auto results_acc = results_buf.get_access<sycl::access::mode::write>(cgh);

    sycl::accessor<uint16_t, 2, sycl::access::mode::read_write, sycl::target::device> local_src(
      sycl::range<2>{16, 16}, cgh);
    sycl::accessor<uint16_t, 2, sycl::access::mode::read_write, sycl::target::device> local_ref(
      sycl::range<2>{16, 16}, cgh);

    auto kern = [&](sycl::nd_item<2> item) {
      const int gid = item.get_group(0) * item.get_group_range(1) + item.get_group(1);
      const int tile_id = item.get_group(2);  // 0-3 for four tiles
      const int lid_x = item.get_local_id(0);
      const int lid_y = item.get_local_id(1);

      if (gid >= num_positions) return;

      const int dy = (gid / (2 * search_range + 1)) - search_range;
      const int dx = (gid % (2 * search_range + 1)) - search_range;

      const int tile_y = (tile_id / 2) * 16;
      const int tile_x = (tile_id % 2) * 16;

      // Load 16x16 tile
      local_src[lid_y][lid_x] = src_acc[tile_y + lid_y][tile_x + lid_x];
      local_ref[lid_y][lid_x] = ref_acc[ref_y + dy + tile_y + lid_y][ref_x + dx + tile_x + lid_x];

      item.barrier(sycl::access::fence_space::local_space);

      uint32_t pixel_sad = sycl::abs(local_src[lid_y][lid_x] - local_ref[lid_y][lid_x]);

      // Reduce to 32x32 result (accumulate across tiles)
      auto group = item.get_group();
      uint32_t tile_sad = sycl::reduce_over_group(group, pixel_sad, sycl::plus<>());

      if (lid_x == 0 && lid_y == 0 && tile_id == 0) {
        // Need atomic add for tile accumulation or use group reduction
        // For simplicity, store each tile result and do final reduction on host
        results_acc[gid * 4 + tile_id] = tile_sad;
      }
    };

    const int num_groups_x = (2 * search_range + 1);
    const int num_groups_y = (2 * search_range + 1);
    cgh.parallel_for<sycl::submit<class SAD32x32Kernel>>(
      sycl::nd_range<3>{4, num_groups_y * 16, num_groups_x * 16},
      sycl::nd_range<3>{1, 16, 16}, kern);
  });

  // Host-side reduction for 32x32 tiles
  q.wait();
  for (int i = 0; i < num_positions; ++i) {
    results[i] = results[i * 4] + results[i * 4 + 1] +
                  results[i * 4 + 2] + results[i * 4 + 3];
  }
}

/// @brief 64x64 SAD kernel (using 16x16 tiles)
void sad64x64(sycl::queue& q, const uint16_t* src, int src_stride,
              const uint16_t* ref, int ref_stride, int ref_x, int ref_y,
              uint32_t* results, const MEParams& params) {
  // Process 64x64 as sixteen 16x16 tiles
  const int block_size = 64;
  const int search_range = params.search_range;
  const int num_positions = (2 * search_range + 1) * (2 * search_range + 1);

  sycl::range<2> local_size{16, 16};

  sycl::buffer<uint16_t, 2> src_buf((uint16_t*)src, sycl::range<2>{block_size, block_size},
                                     {src_stride, 1});
  sycl::buffer<uint16_t, 2> ref_buf((uint16_t*)ref,
                                     sycl::range<2>{block_size + 2*search_range, block_size + 2*search_range},
                                     {ref_stride, 1});
  sycl::buffer<uint32_t, 1> results_buf(results, num_positions * 16);  // 16 tiles per position

  q.submit([&](sycl::handler& cgh) {
    auto src_acc = src_buf.get_access<sycl::access::mode::read>(cgh);
    auto ref_acc = ref_buf.get_access<sycl::access::mode::read>(cgh);
    auto results_acc = results_buf.get_access<sycl::access::mode::write>(cgh);

    sycl::accessor<uint16_t, 2, sycl::access::mode::read_write, sycl::target::device> local_src(
      sycl::range<2>{16, 16}, cgh);
    sycl::accessor<uint16_t, 2, sycl::access::mode::read_write, sycl::target::device> local_ref(
      sycl::range<2>{16, 16}, cgh);

    auto kern = [&](sycl::nd_item<2> item) {
      const int gid = item.get_group(0) * item.get_group_range(1) + item.get_group(1);
      const int tile_id = item.get_group(2);  // 0-15 for sixteen tiles
      const int lid_x = item.get_local_id(0);
      const int lid_y = item.get_local_id(1);

      if (gid >= num_positions) return;

      const int dy = (gid / (2 * search_range + 1)) - search_range;
      const int dx = (gid % (2 * search_range + 1)) - search_range;

      const int tile_y = (tile_id / 4) * 16;
      const int tile_x = (tile_id % 4) * 16;

      local_src[lid_y][lid_x] = src_acc[tile_y + lid_y][tile_x + lid_x];
      local_ref[lid_y][lid_x] = ref_acc[ref_y + dy + tile_y + lid_y][ref_x + dx + tile_x + lid_x];

      item.barrier(sycl::access::fence_space::local_space);

      uint32_t pixel_sad = sycl::abs(local_src[lid_y][lid_x] - local_ref[lid_y][lid_x]);

      auto group = item.get_group();
      uint32_t tile_sad = sycl::reduce_over_group(group, pixel_sad, sycl::plus<>());

      if (lid_x == 0 && lid_y == 0) {
        results_acc[gid * 16 + tile_id] = tile_sad;
      }
    };

    const int num_groups_x = (2 * search_range + 1);
    const int num_groups_y = (2 * search_range + 1);
    cgh.parallel_for<sycl::submit<class SAD64x64Kernel>>(
      sycl::nd_range<3>{16, num_groups_y * 16, num_groups_x * 16},
      sycl::nd_range<3>{1, 16, 16}, kern);
  });

  // Host-side reduction for 64x64 tiles
  q.wait();
  for (int i = 0; i < num_positions; ++i) {
    uint32_t total = 0;
    for (int t = 0; t < 16; ++t) {
      total += results[i * 16 + t];
    }
    results[i] = total;
  }
}

// ============================================================================
// Rectangular Block SAD Kernels
// ============================================================================

void sad4x8(sycl::queue& q, const uint16_t* src, int src_stride,
            const uint16_t* ref, int ref_stride, int ref_x, int ref_y,
            uint32_t* results, const MEParams& params) {
  const int width = 4, height = 8;
  const int search_range = params.search_range;
  const int num_positions = (2 * search_range + 1) * (2 * search_range + 1);

  sycl::buffer<uint16_t, 1> src_buf(src, src_stride * height);
  sycl::buffer<uint16_t, 1> ref_buf(ref, ref_stride * (height + 2 * search_range));
  sycl::buffer<uint32_t, 1> results_buf(results, num_positions);

  q.submit([&](sycl::handler& cgh) {
    auto src_acc = src_buf.get_access<sycl::access::mode::read>(cgh);
    auto ref_acc = ref_buf.get_access<sycl::access::mode::read>(cgh);
    auto results_acc = results_buf.get_access<sycl::access::mode::write>(cgh);

    sycl::accessor<uint16_t, 1, sycl::access::mode::read_write, sycl::target::device> local_src(
      sycl::range<1>(width * height), cgh);
    sycl::accessor<uint16_t, 1, sycl::access::mode::read_write, sycl::target::device> local_ref(
      sycl::range<1>(width * height), cgh);

    auto kern = [&](sycl::nd_item<1> item) {
      const int gid = item.get_global_id(0);
      const int lid = item.get_local_id(0);
      const int group_size = item.get_local_range(0);

      if (gid >= num_positions) return;

      const int dy = (gid / (2 * search_range + 1)) - search_range;
      const int dx = (gid % (2 * search_range + 1)) - search_range;

      for (int i = lid; i < width * height; i += group_size) {
        const int y = i / width;
        const int x = i % width;
        local_src[i] = src_acc[y * src_stride + x];
        local_ref[i] = ref_acc[(ref_y + dy + y) * ref_stride + (ref_x + dx + x)];
      }

      item.barrier(sycl::access::fence_space::local_space);

      uint32_t partial_sad = 0;
      for (int i = lid; i < width * height; i += group_size) {
        partial_sad += sycl::abs(local_src[i] - local_ref[i]);
      }

      auto group_sad = sycl::reduce_over_group(item.get_group(), partial_sad, sycl::plus<>());
      if (lid == 0) {
        results_acc[gid] = group_sad;
      }
    };

    const int local_size = 32;
    const int global_size = ((num_positions + local_size - 1) / local_size) * local_size;
    cgh.parallel_for<class SAD4x8Kernel>(sycl::nd_range<1>(global_size, local_size), kern);
  });
}

void sad8x4(sycl::queue& q, const uint16_t* src, int src_stride,
            const uint16_t* ref, int ref_stride, int ref_x, int ref_y,
            uint32_t* results, const MEParams& params) {
  const int width = 8, height = 4;
  const int search_range = params.search_range;
  const int num_positions = (2 * search_range + 1) * (2 * search_range + 1);

  sycl::buffer<uint16_t, 1> src_buf(src, src_stride * height);
  sycl::buffer<uint16_t, 1> ref_buf(ref, ref_stride * (height + 2 * search_range));
  sycl::buffer<uint32_t, 1> results_buf(results, num_positions);

  q.submit([&](sycl::handler& cgh) {
    auto src_acc = src_buf.get_access<sycl::access::mode::read>(cgh);
    auto ref_acc = ref_buf.get_access<sycl::access::mode::read>(cgh);
    auto results_acc = results_buf.get_access<sycl::access::mode::write>(cgh);

    sycl::accessor<uint16_t, 1, sycl::access::mode::read_write, sycl::target::device> local_src(
      sycl::range<1>(width * height), cgh);
    sycl::accessor<uint16_t, 1, sycl::access::mode::read_write, sycl::target::device> local_ref(
      sycl::range<1>(width * height), cgh);

    auto kern = [&](sycl::nd_item<1> item) {
      const int gid = item.get_global_id(0);
      const int lid = item.get_local_id(0);
      const int group_size = item.get_local_range(0);

      if (gid >= num_positions) return;

      const int dy = (gid / (2 * search_range + 1)) - search_range;
      const int dx = (gid % (2 * search_range + 1)) - search_range;

      for (int i = lid; i < width * height; i += group_size) {
        const int y = i / width;
        const int x = i % width;
        local_src[i] = src_acc[y * src_stride + x];
        local_ref[i] = ref_acc[(ref_y + dy + y) * ref_stride + (ref_x + dx + x)];
      }

      item.barrier(sycl::access::fence_space::local_space);

      uint32_t partial_sad = 0;
      for (int i = lid; i < width * height; i += group_size) {
        partial_sad += sycl::abs(local_src[i] - local_ref[i]);
      }

      auto group_sad = sycl::reduce_over_group(item.get_group(), partial_sad, sycl::plus<>());
      if (lid == 0) {
        results_acc[gid] = group_sad;
      }
    };

    const int local_size = 32;
    const int global_size = ((num_positions + local_size - 1) / local_size) * local_size;
    cgh.parallel_for<class SAD8x4Kernel>(sycl::nd_range<1>(global_size, local_size), kern);
  });
}

void sad8x16(sycl::queue& q, const uint16_t* src, int src_stride,
             const uint16_t* ref, int ref_stride, int ref_x, int ref_y,
             uint32_t* results, const MEParams& params) {
  const int width = 8, height = 16;
  const int search_range = params.search_range;
  const int num_positions = (2 * search_range + 1) * (2 * search_range + 1);

  sycl::range<2> local_size{8, 16};

  sycl::buffer<uint16_t, 2> src_buf((uint16_t*)src, sycl::range<2>{height, width},
                                     {src_stride, 1});
  sycl::buffer<uint16_t, 2> ref_buf((uint16_t*)ref,
                                     sycl::range<2>{height + 2*search_range, width + 2*search_range},
                                     {ref_stride, 1});
  sycl::buffer<uint32_t, 1> results_buf(results, num_positions);

  q.submit([&](sycl::handler& cgh) {
    auto src_acc = src_buf.get_access<sycl::access::mode::read>(cgh);
    auto ref_acc = ref_buf.get_access<sycl::access::mode::read>(cgh);
    auto results_acc = results_buf.get_access<sycl::access::mode::write>(cgh);

    sycl::accessor<uint16_t, 2, sycl::access::mode::read_write, sycl::target::device> local_src(
      sycl::range<2>{height, width}, cgh);
    sycl::accessor<uint16_t, 2, sycl::access::mode::read_write, sycl::target::device> local_ref(
      sycl::range<2>{height, width}, cgh);

    auto kern = [&](sycl::nd_item<2> item) {
      const int gid = item.get_group(0) * item.get_group_range(1) + item.get_group(1);
      const int lid_x = item.get_local_id(0);
      const int lid_y = item.get_local_id(1);

      if (gid >= num_positions) return;

      const int dy = (gid / (2 * search_range + 1)) - search_range;
      const int dx = (gid % (2 * search_range + 1)) - search_range;

      local_src[lid_y][lid_x] = src_acc[lid_y][lid_x];
      local_ref[lid_y][lid_x] = ref_acc[ref_y + dy + lid_y][ref_x + dx + lid_x];

      item.barrier(sycl::access::fence_space::local_space);

      uint32_t pixel_sad = sycl::abs(local_src[lid_y][lid_x] - local_ref[lid_y][lid_x]);
      auto group = item.get_group();
      uint32_t tile_sad = sycl::reduce_over_group(group, pixel_sad, sycl::plus<>());

      if (lid_x == 0 && lid_y == 0) {
        results_acc[gid] = tile_sad;
      }
    };

    const int num_groups_x = (2 * search_range + 1);
    const int num_groups_y = (2 * search_range + 1);
    cgh.parallel_for<class SAD8x16Kernel>(
      sycl::nd_range<2>{num_groups_y * 16, num_groups_x * 8},
      sycl::nd_range<2>{16, 8}, kern);
  });
}

void sad16x8(sycl::queue& q, const uint16_t* src, int src_stride,
             const uint16_t* ref, int ref_stride, int ref_x, int ref_y,
             uint32_t* results, const MEParams& params) {
  const int width = 16, height = 8;
  const int search_range = params.search_range;
  const int num_positions = (2 * search_range + 1) * (2 * search_range + 1);

  sycl::range<2> local_size{16, 8};

  sycl::buffer<uint16_t, 2> src_buf((uint16_t*)src, sycl::range<2>{height, width},
                                     {src_stride, 1});
  sycl::buffer<uint16_t, 2> ref_buf((uint16_t*)ref,
                                     sycl::range<2>{height + 2*search_range, width + 2*search_range},
                                     {ref_stride, 1});
  sycl::buffer<uint32_t, 1> results_buf(results, num_positions);

  q.submit([&](sycl::handler& cgh) {
    auto src_acc = src_buf.get_access<sycl::access::mode::read>(cgh);
    auto ref_acc = ref_buf.get_access<sycl::access::mode::read>(cgh);
    auto results_acc = results_buf.get_access<sycl::access::mode::write>(cgh);

    sycl::accessor<uint16_t, 2, sycl::access::mode::read_write, sycl::target::device> local_src(
      sycl::range<2>{height, width}, cgh);
    sycl::accessor<uint16_t, 2, sycl::access::mode::read_write, sycl::target::device> local_ref(
      sycl::range<2>{height, width}, cgh);

    auto kern = [&](sycl::nd_item<2> item) {
      const int gid = item.get_group(0) * item.get_group_range(1) + item.get_group(1);
      const int lid_x = item.get_local_id(0);
      const int lid_y = item.get_local_id(1);

      if (gid >= num_positions) return;

      const int dy = (gid / (2 * search_range + 1)) - search_range;
      const int dx = (gid % (2 * search_range + 1)) - search_range;

      local_src[lid_y][lid_x] = src_acc[lid_y][lid_x];
      local_ref[lid_y][lid_x] = ref_acc[ref_y + dy + lid_y][ref_x + dx + lid_x];

      item.barrier(sycl::access::fence_space::local_space);

      uint32_t pixel_sad = sycl::abs(local_src[lid_y][lid_x] - local_ref[lid_y][lid_x]);
      auto group = item.get_group();
      uint32_t tile_sad = sycl::reduce_over_group(group, pixel_sad, sycl::plus<>());

      if (lid_x == 0 && lid_y == 0) {
        results_acc[gid] = tile_sad;
      }
    };

    const int num_groups_x = (2 * search_range + 1);
    const int num_groups_y = (2 * search_range + 1);
    cgh.parallel_for<class SAD16x8Kernel>(
      sycl::nd_range<2>{num_groups_y * 8, num_groups_x * 16},
      sycl::nd_range<2>{8, 16}, kern);
  });
}

void sad16x32(sycl::queue& q, const uint16_t* src, int src_stride,
              const uint16_t* ref, int ref_stride, int ref_x, int ref_y,
              uint32_t* results, const MEParams& params) {
  const int width = 16, height = 32;
  const int search_range = params.search_range;
  const int num_positions = (2 * search_range + 1) * (2 * search_range + 1);

  // Process as two 16x16 tiles
  sycl::range<2> local_size{16, 16};

  sycl::buffer<uint16_t, 2> src_buf((uint16_t*)src, sycl::range<2>{height, width},
                                     {src_stride, 1});
  sycl::buffer<uint16_t, 2> ref_buf((uint16_t*)ref,
                                     sycl::range<2>{height + 2*search_range, width + 2*search_range},
                                     {ref_stride, 1});
  sycl::buffer<uint32_t, 1> results_buf(results, num_positions * 2);

  q.submit([&](sycl::handler& cgh) {
    auto src_acc = src_buf.get_access<sycl::access::mode::read>(cgh);
    auto ref_acc = ref_buf.get_access<sycl::access::mode::read>(cgh);
    auto results_acc = results_buf.get_access<sycl::access::mode::write>(cgh);

    sycl::accessor<uint16_t, 2, sycl::access::mode::read_write, sycl::target::device> local_src(
      sycl::range<2>{16, 16}, cgh);
    sycl::accessor<uint16_t, 2, sycl::access::mode::read_write, sycl::target::device> local_ref(
      sycl::range<2>{16, 16}, cgh);

    auto kern = [&](sycl::nd_item<2> item) {
      const int gid = item.get_group(0) * item.get_group_range(1) + item.get_group(1);
      const int tile_id = item.get_group(2);
      const int lid_x = item.get_local_id(0);
      const int lid_y = item.get_local_id(1);

      if (gid >= num_positions) return;

      const int dy = (gid / (2 * search_range + 1)) - search_range;
      const int dx = (gid % (2 * search_range + 1)) - search_range;

      const int tile_y = tile_id * 16;

      local_src[lid_y][lid_x] = src_acc[tile_y + lid_y][lid_x];
      local_ref[lid_y][lid_x] = ref_acc[ref_y + dy + tile_y + lid_y][ref_x + dx + lid_x];

      item.barrier(sycl::access::fence_space::local_space);

      uint32_t pixel_sad = sycl::abs(local_src[lid_y][lid_x] - local_ref[lid_y][lid_x]);
      auto group = item.get_group();
      uint32_t tile_sad = sycl::reduce_over_group(group, pixel_sad, sycl::plus<>());

      if (lid_x == 0 && lid_y == 0) {
        results_acc[gid * 2 + tile_id] = tile_sad;
      }
    };

    const int num_groups_x = (2 * search_range + 1);
    const int num_groups_y = (2 * search_range + 1);
    cgh.parallel_for<class SAD16x32Kernel>(
      sycl::nd_range<3>{2, num_groups_y * 16, num_groups_x * 16},
      sycl::nd_range<3>{1, 16, 16}, kern);
  });

  q.wait();
  for (int i = 0; i < num_positions; ++i) {
    results[i] = results[i * 2] + results[i * 2 + 1];
  }
}

void sad32x16(sycl::queue& q, const uint16_t* src, int src_stride,
              const uint16_t* ref, int ref_stride, int ref_x, int ref_y,
              uint32_t* results, const MEParams& params) {
  const int width = 32, height = 16;
  const int search_range = params.search_range;
  const int num_positions = (2 * search_range + 1) * (2 * search_range + 1);

  // Process as two 16x16 tiles
  sycl::range<2> local_size{16, 16};

  sycl::buffer<uint16_t, 2> src_buf((uint16_t*)src, sycl::range<2>{height, width},
                                     {src_stride, 1});
  sycl::buffer<uint16_t, 2> ref_buf((uint16_t*)ref,
                                     sycl::range<2>{height + 2*search_range, width + 2*search_range},
                                     {ref_stride, 1});
  sycl::buffer<uint32_t, 1> results_buf(results, num_positions * 2);

  q.submit([&](sycl::handler& cgh) {
    auto src_acc = src_buf.get_access<sycl::access::mode::read>(cgh);
    auto ref_acc = ref_buf.get_access<sycl::access::mode::read>(cgh);
    auto results_acc = results_buf.get_access<sycl::access::mode::write>(cgh);

    sycl::accessor<uint16_t, 2, sycl::access::mode::read_write, sycl::target::device> local_src(
      sycl::range<2>{16, 16}, cgh);
    sycl::accessor<uint16_t, 2, sycl::access::mode::read_write, sycl::target::device> local_ref(
      sycl::range<2>{16, 16}, cgh);

    auto kern = [&](sycl::nd_item<2> item) {
      const int gid = item.get_group(0) * item.get_group_range(1) + item.get_group(1);
      const int tile_id = item.get_group(2);
      const int lid_x = item.get_local_id(0);
      const int lid_y = item.get_local_id(1);

      if (gid >= num_positions) return;

      const int dy = (gid / (2 * search_range + 1)) - search_range;
      const int dx = (gid % (2 * search_range + 1)) - search_range;

      const int tile_x = tile_id * 16;

      local_src[lid_y][lid_x] = src_acc[lid_y][tile_x + lid_x];
      local_ref[lid_y][lid_x] = ref_acc[ref_y + dy + lid_y][ref_x + dx + tile_x + lid_x];

      item.barrier(sycl::access::fence_space::local_space);

      uint32_t pixel_sad = sycl::abs(local_src[lid_y][lid_x] - local_ref[lid_y][lid_x]);
      auto group = item.get_group();
      uint32_t tile_sad = sycl::reduce_over_group(group, pixel_sad, sycl::plus<>());

      if (lid_x == 0 && lid_y == 0) {
        results_acc[gid * 2 + tile_id] = tile_sad;
      }
    };

    const int num_groups_x = (2 * search_range + 1);
    const int num_groups_y = (2 * search_range + 1);
    cgh.parallel_for<class SAD32x16Kernel>(
      sycl::nd_range<3>{2, num_groups_y * 16, num_groups_x * 16},
      sycl::nd_range<3>{1, 16, 16}, kern);
  });

  q.wait();
  for (int i = 0; i < num_positions; ++i) {
    results[i] = results[i * 2] + results[i * 2 + 1];
  }
}

// ============================================================================
// Multi-Candidate SAD
// ============================================================================

void sad_multi_candidate(sycl::queue& q, const uint16_t* src, int src_stride,
                         const uint16_t* ref, int ref_stride,
                         const int2* candidates, int num_candidates,
                         uint32_t* results, int width, int height) {
  sycl::buffer<int2> cand_buf(candidates, num_candidates);
  sycl::buffer<uint32_t> results_buf(results, num_candidates);

  q.submit([&](sycl::handler& cgh) {
    auto cand_acc = cand_buf.get_access<sycl::access::mode::read>(cgh);
    auto results_acc = results_buf.get_access<sycl::access::mode::write>(cgh);

    auto kern = [=](sycl::id<1> gid) {
      if (gid >= num_candidates) return;

      const int2 offset = cand_acc[gid];
      uint32_t sad = 0;

      for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
          int src_idx = y * src_stride + x;
          int ref_idx = (y + offset.y()) * ref_stride + (x + offset.x());
          sad += sycl::abs(src[src_idx] - ref[ref_idx]);
        }
      }

      results_acc[gid] = sad;
    };

    cgh.parallel_for<class SADMultiCandidate>(sycl::range<1>(num_candidates), kern);
  });
}

void sad_diamond_4way(sycl::queue& q, const uint16_t* src, int src_stride,
                      const uint16_t* ref, int ref_stride,
                      int center_x, int center_y, uint32_t* sad_out,
                      int width, int height) {
  // Diamond pattern: (0,-1), (0,1), (-1,0), (1,0)
  const int2 offsets[4] = {{0, -1}, {0, 1}, {-1, 0}, {1, 0}};

  sycl::buffer<int2> offset_buf(offsets, 4);
  sycl::buffer<uint32_t> sad_buf(sad_out, 4);

  q.submit([&](sycl::handler& cgh) {
    auto offset_acc = offset_buf.get_access<sycl::access::mode::read>(cgh);
    auto sad_acc = sad_buf.get_access<sycl::access::mode::write>(cgh);

    auto kern = [=](sycl::id<1> gid) {
      const int2 offset = offset_acc[gid];
      const int ref_start_y = center_y + offset.y();
      const int ref_start_x = center_x + offset.x();

      uint32_t sad = 0;
      for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
          int src_idx = y * src_stride + x;
          int ref_idx = (ref_start_y + y) * ref_stride + (ref_start_x + x);
          sad += sycl::abs(src[src_idx] - ref[ref_idx]);
        }
      }

      sad_acc[gid] = sad;
    };

    cgh.parallel_for<class SADDiamond4Way>(sycl::range<1>(4), kern);
  });
}

void sad_diamond_8way(sycl::queue& q, const uint16_t* src, int src_stride,
                      const uint16_t* ref, int ref_stride,
                      int center_x, int center_y, uint32_t* sad_out,
                      int width, int height) {
  // 8-way pattern: (0,-1), (0,1), (-1,0), (1,0), (-1,-1), (-1,1), (1,-1), (1,1)
  const int2 offsets[8] = {{0, -1}, {0, 1}, {-1, 0}, {1, 0},
                           {-1, -1}, {-1, 1}, {1, -1}, {1, 1}};

  sycl::buffer<int2> offset_buf(offsets, 8);
  sycl::buffer<uint32_t> sad_buf(sad_out, 8);

  q.submit([&](sycl::handler& cgh) {
    auto offset_acc = offset_buf.get_access<sycl::access::mode::read>(cgh);
    auto sad_acc = sad_buf.get_access<sycl::access::mode::write>(cgh);

    auto kern = [=](sycl::id<1> gid) {
      const int2 offset = offset_acc[gid];
      const int ref_start_y = center_y + offset.y();
      const int ref_start_x = center_x + offset.x();

      uint32_t sad = 0;
      for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
          int src_idx = y * src_stride + x;
          int ref_idx = (ref_start_y + y) * ref_stride + (ref_start_x + x);
          sad += sycl::abs(src[src_idx] - ref[ref_idx]);
        }
      }

      sad_acc[gid] = sad;
    };

    cgh.parallel_for<class SADDiamond8Way>(sycl::range<1>(8), kern);
  });
}

// ============================================================================
// Full Search Motion Estimation
// ============================================================================

MVResult full_search_me(sycl::queue& q, const uint16_t* src, int src_stride,
                        const uint16_t* ref, int ref_stride,
                        int ref_x, int ref_y, const MEParams& params) {
  int width, height;
  get_block_dimensions(params.bsize, width, height);

  const int num_positions = (2 * params.search_range + 1) * (2 * params.search_range + 1);
  std::vector<uint32_t> sad_results(num_positions);

  // Dispatch to appropriate SAD function
  if (width == 4 && height == 4) {
    sad4x4(q, src, src_stride, ref, ref_stride, ref_x, ref_y,
           sad_results.data(), params);
  } else if (width == 8 && height == 8) {
    sad8x8(q, src, src_stride, ref, ref_stride, ref_x, ref_y,
           sad_results.data(), params);
  } else if (width == 16 && height == 16) {
    sad16x16(q, src, src_stride, ref, ref_stride, ref_x, ref_y,
             sad_results.data(), params);
  } else if (width == 32 && height == 32) {
    sad32x32(q, src, src_stride, ref, ref_stride, ref_x, ref_y,
             sad_results.data(), params);
  } else if (width == 64 && height == 64) {
    sad64x64(q, src, src_stride, ref, ref_stride, ref_x, ref_y,
             sad_results.data(), params);
  } else {
    // Fallback to generic implementation for other sizes
    for (int dy = -params.search_range; dy <= params.search_range; ++dy) {
      for (int dx = -params.search_range; dx <= params.search_range; ++dx) {
        const int idx = (dy + params.search_range) * (2 * params.search_range + 1) +
                        (dx + params.search_range);
        uint32_t sad = 0;
        for (int y = 0; y < height; ++y) {
          for (int x = 0; x < width; ++x) {
            sad += std::abs(src[y * src_stride + x] -
                           ref[(ref_y + dy + y) * ref_stride + (ref_x + dx + x)]);
          }
        }
        sad_results[idx] = sad;
      }
    }
  }

  // Find minimum SAD
  uint32_t min_sad = sad_results[0];
  int best_idx = 0;
  for (int i = 1; i < num_positions; ++i) {
    if (sad_results[i] < min_sad) {
      min_sad = sad_results[i];
      best_idx = i;
    }
  }

  // Convert index to MV
  const int dy = (best_idx / (2 * params.search_range + 1)) - params.search_range;
  const int dx = (best_idx % (2 * params.search_range + 1)) - params.search_range;

  MVResult result;
  result.mv_x = params.start_x + dx;
  result.mv_y = params.start_y + dy;
  result.sad = min_sad;

  return result;
}

// ============================================================================
// Diamond Search Motion Estimation
// ============================================================================

MVResult diamond_search_me(sycl::queue& q, const uint16_t* src, int src_stride,
                           const uint16_t* ref, int ref_stride,
                           const MVResult& start_mv, const MEParams& params) {
  int width, height;
  get_block_dimensions(params.bsize, width, height);

  // Convert 1/8 pel units to full-pel
  const int start_x_full = start_mv.mv_x / 8;
  const int start_y_full = start_mv.mv_y / 8;

  MVResult best_mv = start_mv;
  uint32_t best_sad = start_mv.sad;

  // Iterative diamond search
  constexpr int max_iterations = 16;
  int iteration = 0;

  while (iteration < max_iterations) {
    uint32_t sad_4way[4];

    sad_diamond_4way(q, src, src_stride, ref, ref_stride,
                     start_x_full + best_mv.mv_x / 8,
                     start_y_full + best_mv.mv_y / 8,
                     sad_4way, width, height);

    q.wait();

    // Find best in diamond pattern
    int best_direction = -1;
    uint32_t min_sad = best_sad;

    for (int i = 0; i < 4; ++i) {
      if (sad_4way[i] < min_sad) {
        min_sad = sad_4way[i];
        best_direction = i;
      }
    }

    if (best_direction == -1) {
      // Local minimum found
      break;
    }

    // Update MV
    constexpr int diamond_offsets[4][2] = {{0, -1}, {0, 1}, {-1, 0}, {1, 0}};
    best_mv.mv_x += diamond_offsets[best_direction][0] * 8;
    best_mv.mv_y += diamond_offsets[best_direction][1] * 8;
    best_mv.sad = min_sad;
    best_sad = min_sad;

    ++iteration;
  }

  return best_mv;
}

// ============================================================================
// Hierarchical Motion Estimation
// ============================================================================

MVResult hierarchical_me(sycl::queue& q,
                        const uint16_t* src, int src_stride,
                        const uint16_t* ref, int ref_stride,
                        const uint16_t* src_ds, int src_ds_stride,
                        const uint16_t* ref_ds, int ref_ds_stride,
                        const MEParams& params) {
  int width, height;
  get_block_dimensions(params.bsize, width, height);

  const int width_ds = (width + 1) / 2;
  const int height_ds = (height + 1) / 2;

  // Stage 1: Coarse search at 2x downsampled resolution
  MEParams coarse_params = params;
  coarse_params.search_range = params.search_range / 2;  // Reduced range at coarse level

  MVResult coarse_mv = full_search_me(q, src_ds, src_ds_stride,
                                       ref_ds, ref_ds_stride,
                                       0, 0, coarse_params);

  // Stage 2: Refine at full resolution
  MEParams fine_params = params;
  fine_params.start_x = coarse_mv.mv_x * 2;  // Scale up to full resolution
  fine_params.start_y = coarse_mv.mv_y * 2;
  fine_params.search_range = 2;  // Small refinement window

  MVResult final_mv = full_search_me(q, src, src_stride,
                                      ref, ref_stride,
                                      0, 0, fine_params);

  return final_mv;
}

// ============================================================================
// Sub-pixel Motion Estimation
// ============================================================================

MVResult subpel_halfpel_me(sycl::queue& q, const uint16_t* src, int src_stride,
                           const uint16_t* ref, int ref_stride,
                           const MVResult& start_mv, const MEParams& params) {
  int width, height;
  get_block_dimensions(params.bsize, width, height);

  // Half-pel positions: (0,0), (0,1), (1,0), (1,1) in half-pel units
  const int2 half_pel_offsets[4] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
  uint32_t sad_halfpel[4];

  // Assume ref contains interpolated half-pel values
  // Each half-pel position is stored consecutively
  for (int i = 0; i < 4; ++i) {
    uint32_t sad = 0;
    const int ref_idx_base = (half_pel_offsets[i].y() * ref_stride + half_pel_offsets[i].x());

    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        sad += std::abs(src[y * src_stride + x] -
                       ref[(y + ref_idx_base) * ref_stride + (x + ref_idx_base)]);
      }
    }
    sad_halfpel[i] = sad;
  }

  // Find best half-pel position
  int best_halfpel = 0;
  uint32_t min_sad = sad_halfpel[0];
  for (int i = 1; i < 4; ++i) {
    if (sad_halfpel[i] < min_sad) {
      min_sad = sad_halfpel[i];
      best_halfpel = i;
    }
  }

  MVResult result;
  result.mv_x = start_mv.mv_x + half_pel_offsets[best_halfpel].x() * 4;  // Half-pel = 1/4 of 1/8 pel
  result.mv_y = start_mv.mv_y + half_pel_offsets[best_halfpel].y() * 4;
  result.sad = min_sad;

  return result;
}

MVResult subpel_quarterpel_me(sycl::queue& q, const uint16_t* src, int src_stride,
                              const uint16_t* ref, int ref_stride,
                              const MVResult& start_mv, const MEParams& params) {
  int width, height;
  get_block_dimensions(params.bsize, width, height);

  // Quarter-pel positions around best half-pel position
  const int2 qpel_offsets[8] = {{0, 0}, {0, 1}, {1, 0}, {1, 1},
                                 {0, 2}, {2, 0}, {2, 2}, {1, 2}};
  uint32_t sad_qpel[8];

  for (int i = 0; i < 8; ++i) {
    uint32_t sad = 0;
    const int ref_idx_base = (qpel_offsets[i].y() * ref_stride + qpel_offsets[i].x());

    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        sad += std::abs(src[y * src_stride + x] -
                       ref[(y + ref_idx_base) * ref_stride + (x + ref_idx_base)]);
      }
    }
    sad_qpel[i] = sad;
  }

  // Find best quarter-pel position
  int best_qpel = 0;
  uint32_t min_sad = sad_qpel[0];
  for (int i = 1; i < 8; ++i) {
    if (sad_qpel[i] < min_sad) {
      min_sad = sad_qpel[i];
      best_qpel = i;
    }
  }

  MVResult result;
  result.mv_x = start_mv.mv_x + qpel_offsets[best_qpel].x() * 2;  // Quarter-pel = 1/2 of 1/8 pel
  result.mv_y = start_mv.mv_y + qpel_offsets[best_qpel].y() * 2;
  result.sad = min_sad;

  return result;
}

// ============================================================================
// Batch Motion Estimation
// ============================================================================

void batch_full_search_me(sycl::queue& q,
                          const uint16_t* src, int src_stride,
                          const uint16_t* ref, int ref_stride,
                          const int2* block_origins, int num_blocks,
                          const MEParams& params, MVResult* results) {
  for (int i = 0; i < num_blocks; ++i) {
    const int2 origin = block_origins[i];
    MEParams block_params = params;
    block_params.start_x = origin.x();
    block_params.start_y = origin.y();

    results[i] = full_search_me(q, src + origin.y() * src_stride + origin.x(),
                                 src_stride, ref, ref_stride,
                                 origin.x(), origin.y(), block_params);
  }
}

// ============================================================================
// Utility Functions
// ============================================================================

SADFunc get_sad_function(BLOCK_SIZE bsize) {
  switch (bsize) {
    case BLOCK_4X4: return sad4x4;
    case BLOCK_4X8: return sad4x8;
    case BLOCK_8X4: return sad8x4;
    case BLOCK_8X8: return sad8x8;
    case BLOCK_8X16: return sad8x16;
    case BLOCK_16X8: return sad16x8;
    case BLOCK_16X16: return sad16x16;
    case BLOCK_16X32: return sad16x32;
    case BLOCK_32X16: return sad32x16;
    case BLOCK_32X32: return sad32x32;
    case BLOCK_64X64: return sad64x64;
    default: return nullptr;
  }
}

void get_block_dimensions(BLOCK_SIZE bsize, int& width, int& height) {
  width = block_size_wide[bsize];
  height = block_size_high[bsize];
}

}  // namespace sycl
}  // namespace avm
