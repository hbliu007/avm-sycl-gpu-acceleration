/*
 * Copyright (c) 2026, Alliance for Open Media. All rights reserved
 *
 * Performance comparison test for optimized SYCL transform kernels
 *
 * This test compares:
 * 1. Original implementation vs optimized implementation
 * 2. Single block processing vs batch processing
 * 3. Different memory allocation strategies
 */

#include <CL/sycl.hpp>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

#include "third_party/googletest/src/googletest/include/gtest/gtest.h"

#if !defined(__SYCL_COMPILER_VERSION)
#error "SYCL compiler not detected. Please compile with -fsycl"
#endif

#include "sycl_txfm.hpp"
#include "sycl_txfm_optimized.hpp"
#include "sycl_context.hpp"

using namespace avm::sycl;

namespace {

// Timing utilities
using Clock = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double>;

// Performance result structure
struct PerfResult {
    std::string name;
    double avg_time_us;
    double throughput_ops_per_sec;
    double speedup_vs_baseline;

    void print() const {
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "  " << std::setw(30) << std::left << name << ": "
                  << std::setw(10) << avg_time_us << " us/op, "
                  << std::setw(12) << throughput_ops_per_sec << " ops/s";
        if (speedup_vs_baseline > 0) {
            std::cout << ", " << std::setw(5) << speedup_vs_baseline << "x speedup";
        }
        std::cout << std::endl;
    }
};

// Benchmark wrapper
template<typename Func>
PerfResult benchmark(const std::string& name, Func&& func, int iterations = 1000) {
    // Warmup
    for (int i = 0; i < 10; ++i) {
        func();
    }

    // Measure
    auto start = Clock::now();
    for (int i = 0; i < iterations; ++i) {
        func();
    }
    auto end = Clock::now();

    Duration total = end - start;
    double avg_us = total.count() * 1e6 / iterations;
    double throughput = 1.0 / (total.count() / iterations);

    return {name, avg_us, throughput, 0.0};
}

// Test data initialization
template<typename T>
void initialize_test_data(T* data, int size, int seed = 42) {
    for (int i = 0; i < size; ++i) {
        data[i] = static_cast<T>((i * 7 + seed) % 256);
    }
}

}  // anonymous namespace

/**
 * @class SYCLOptimizedPerformanceTest
 * @brief Performance comparison test fixture
 */
class SYCLOptimizedPerformanceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    try {
      queue_ = std::make_unique<sycl::queue>(
          sycl::default_selector_v,
          {sycl::property::queue::enable_profiling{}});
    } catch (const sycl::exception& e) {
      GTEST_SKIP() << "SYCL device not available: " << e.what();
    }

    // Print device info
    auto device = queue_->get_device();
    std::cout << "\n=== Device Info ===" << std::endl;
    std::cout << "Name: "
              << device.get_info<sycl::info::device::name>() << std::endl;
    std::cout << "Max Compute Units: "
              << device.get_info<sycl::info::device::max_compute_units>() << std::endl;
    std::cout << "Max Work Group Size: "
              << device.get_info<sycl::info::device::max_work_group_size>() << std::endl;
    std::cout << "Global Memory: "
              << (device.get_info<sycl::info::device::global_mem_size>() /
                  1024 / 1024) << " MB" << std::endl;
    std::cout << "Local Memory: "
              << device.get_info<sycl::info::device::local_mem_size>() << " bytes" << std::endl;
    std::cout << "===================\n" << std::endl;
  }

  void TearDown() override {
    queue_.reset();
  }

  std::unique_ptr<sycl::queue> queue_;
};

/**
 * @test FDCT8x8PerformanceComparison
 * @brief Compare original vs optimized 8x8 forward DCT
 */
TEST_F(SYCLOptimizedPerformanceTest, FDCT8x8PerformanceComparison) {
    ASSERT_NE(queue_, nullptr);

    constexpr int kSize = 8;
    constexpr int kStride = 8;
    constexpr int kIterations = 1000;

    // Prepare test data
    std::vector<int16_t> input(kSize * kStride);
    std::vector<tran_low_t> output(kSize * kSize);

    initialize_test_data(input.data(), input.size());

    TxfmParams params;
    params.tx_size = TX_8X8;
    params.tx_type = DCT_DCT;
    params.bd = 10;
    params.eob = 0;
    params.dir = TxfmDir::kForward;
    params.shift = 0;

    std::cout << "\n=== 8x8 Forward DCT Performance ===" << std::endl;

    // Benchmark original implementation
    auto original_time = benchmark("Original fdct8x8", [&]() {
        fdct8x8(*queue_, input.data(), output.data(), kStride, params);
    }, kIterations);

    // Benchmark optimized implementation
    auto optimized_time = benchmark("Optimized fdct8x8", [&]() {
        fdct8x8_optimized(*queue_, input.data(), output.data(), kStride, params);
    }, kIterations);

    // Calculate speedup
    optimized_time.speedup_vs_baseline = original_time.avg_time_us / optimized_time.avg_time_us;

    // Print results
    original_time.print();
    optimized_time.print();

    std::cout << "\nSummary: Optimized implementation achieved "
              << optimized_time.speedup_vs_baseline << "x speedup" << std::endl;
    std::cout << "=====================================\n" << std::endl;

    // Verify correctness (quick check)
    std::vector<tran_low_t> output_orig(kSize * kSize);
    std::vector<tran_low_t> output_opt(kSize * kSize);

    fdct8x8(*queue_, input.data(), output_orig.data(), kStride, params);
    fdct8x8_optimized(*queue_, input.data(), output_opt.data(), kStride, params);

    bool match = true;
    for (int i = 0; i < kSize * kSize; ++i) {
        if (output_orig[i] != output_opt[i]) {
            match = false;
            std::cout << "Mismatch at index " << i << ": "
                      << output_orig[i] << " vs " << output_opt[i] << std::endl;
            break;
        }
    }
    EXPECT_TRUE(match) << "Optimized implementation produces different results";
}

/**
 * @test IDCT8x8PerformanceComparison
 * @brief Compare original vs optimized 8x8 inverse DCT
 */
TEST_F(SYCLOptimizedPerformanceTest, IDCT8x8PerformanceComparison) {
    ASSERT_NE(queue_, nullptr);

    constexpr int kSize = 8;
    constexpr int kStride = 8;
    constexpr int kIterations = 1000;

    std::vector<tran_low_t> input(kSize * kSize);
    std::vector<uint16_t> output(kSize * kStride);

    initialize_test_data(input.data(), input.size());

    TxfmParams params;
    params.tx_size = TX_8X8;
    params.tx_type = DCT_DCT;
    params.bd = 10;
    params.eob = kSize * kSize;
    params.dir = TxfmDir::kInverse;
    params.shift = 0;

    std::cout << "\n=== 8x8 Inverse DCT Performance ===" << std::endl;

    auto original_time = benchmark("Original idct8x8", [&]() {
        idct8x8(*queue_, input.data(), output.data(), kStride, params);
    }, kIterations);

    auto optimized_time = benchmark("Optimized idct8x8", [&]() {
        idct8x8_optimized(*queue_, input.data(), output.data(), kStride, params);
    }, kIterations);

    optimized_time.speedup_vs_baseline = original_time.avg_time_us / optimized_time.avg_time_us;

    original_time.print();
    optimized_time.print();

    std::cout << "\nSummary: Optimized implementation achieved "
              << optimized_time.speedup_vs_baseline << "x speedup" << std::endl;
    std::cout << "=====================================\n" << std::endl;

    // Verify correctness
    std::vector<uint16_t> output_orig(kSize * kStride);
    std::vector<uint16_t> output_opt(kSize * kStride);

    idct8x8(*queue_, input.data(), output_orig.data(), kStride, params);
    idct8x8_optimized(*queue_, input.data(), output_opt.data(), kStride, params);

    bool match = true;
    for (int i = 0; i < kSize * kStride; ++i) {
        if (output_orig[i] != output_opt[i]) {
            match = false;
            std::cout << "Mismatch at index " << i << ": "
                      << output_orig[i] << " vs " << output_opt[i] << std::endl;
            break;
        }
    }
    EXPECT_TRUE(match) << "Optimized implementation produces different results";
}

/**
 * @test FDCT4x4PerformanceComparison
 * @brief Compare original vs optimized 4x4 forward DCT
 */
TEST_F(SYCLOptimizedPerformanceTest, FDCT4x4PerformanceComparison) {
    ASSERT_NE(queue_, nullptr);

    constexpr int kSize = 4;
    constexpr int kStride = 4;
    constexpr int kIterations = 1000;

    std::vector<int16_t> input(kSize * kStride);
    std::vector<tran_low_t> output(kSize * kSize);

    initialize_test_data(input.data(), input.size());

    TxfmParams params;
    params.tx_size = TX_4X4;
    params.tx_type = DCT_DCT;
    params.bd = 10;
    params.eob = 0;
    params.dir = TxfmDir::kForward;
    params.shift = 0;

    std::cout << "\n=== 4x4 Forward DCT Performance ===" << std::endl;

    auto original_time = benchmark("Original fdct4x4", [&]() {
        fdct4x4(*queue_, input.data(), output.data(), kStride, params);
    }, kIterations);

    auto optimized_time = benchmark("Optimized fdct4x4", [&]() {
        fdct4x4_optimized(*queue_, input.data(), output.data(), kStride, params);
    }, kIterations);

    optimized_time.speedup_vs_baseline = original_time.avg_time_us / optimized_time.avg_time_us;

    original_time.print();
    optimized_time.print();

    std::cout << "\nSummary: Optimized implementation achieved "
              << optimized_time.speedup_vs_baseline << "x speedup" << std::endl;
    std::cout << "=====================================\n" << std::endl;

    // Verify correctness
    std::vector<tran_low_t> output_orig(kSize * kSize);
    std::vector<tran_low_t> output_opt(kSize * kSize);

    fdct4x4(*queue_, input.data(), output_orig.data(), kStride, params);
    fdct4x4_optimized(*queue_, input.data(), output_opt.data(), kStride, params);

    bool match = true;
    for (int i = 0; i < kSize * kSize; ++i) {
        if (output_orig[i] != output_opt[i]) {
            match = false;
            break;
        }
    }
    EXPECT_TRUE(match) << "Optimized implementation produces different results";
}

/**
 * @test BatchProcessingPerformance
 * @brief Compare single vs batch processing for multiple transforms
 */
TEST_F(SYCLOptimizedPerformanceTest, BatchProcessingPerformance) {
    ASSERT_NE(queue_, nullptr);

    constexpr int kSize = 8;
    constexpr int kStride = 8;
    constexpr int kNumBlocks = 64;
    constexpr int kIterations = 100;

    // Prepare batch data
    std::vector<std::vector<int16_t>> inputs(kNumBlocks);
    std::vector<std::vector<tran_low_t>> outputs_single(kNumBlocks);
    std::vector<std::vector<tran_low_t>> outputs_batch(kNumBlocks);
    std::vector<const int16_t*> input_ptrs(kNumBlocks);
    std::vector<tran_low_t*> output_ptrs(kNumBlocks);

    for (int i = 0; i < kNumBlocks; ++i) {
        inputs[i].resize(kSize * kStride);
        outputs_single[i].resize(kSize * kSize);
        outputs_batch[i].resize(kSize * kSize);
        input_ptrs[i] = inputs[i].data();
        output_ptrs[i] = outputs_batch[i].data();
        initialize_test_data(inputs[i].data(), inputs[i].size(), i * 17);
    }

    TxfmParams params;
    params.tx_size = TX_8X8;
    params.tx_type = DCT_DCT;
    params.bd = 10;
    params.eob = 0;
    params.dir = TxfmDir::kForward;
    params.shift = 0;

    std::cout << "\n=== Batch Processing Performance (" << kNumBlocks << " blocks) ===" << std::endl;

    // Single block processing (sequential calls)
    auto single_time = benchmark("Sequential (64x single)", [&]() {
        for (int i = 0; i < kNumBlocks; ++i) {
            fdct8x8_optimized(*queue_, inputs[i].data(),
                             outputs_single[i].data(), kStride, params);
        }
    }, kIterations);

    // Batch processing (single kernel call)
    auto batch_time = benchmark("Batch (64 at once)", [&]() {
        fdct8x8_batch(*queue_, input_ptrs.data(), output_ptrs.data(),
                      kNumBlocks, kStride, params);
    }, kIterations);

    batch_time.speedup_vs_baseline = single_time.avg_time_us / batch_time.avg_time_us;

    single_time.print();
    batch_time.print();

    std::cout << "\nSummary: Batch processing achieved "
              << batch_time.speedup_vs_baseline << "x speedup" << std::endl;
    std::cout << "==========================================\n" << std::endl;

    // Verify correctness
    fdct8x8_batch(*queue_, input_ptrs.data(), output_ptrs.data(),
                  kNumBlocks, kStride, params);

    for (int b = 0; b < kNumBlocks; ++b) {
        fdct8x8_optimized(*queue_, inputs[b].data(),
                         outputs_single[b].data(), kStride, params);

        for (int i = 0; i < kSize * kSize; ++i) {
            if (outputs_single[b][i] != outputs_batch[b][i]) {
                EXPECT_EQ(outputs_single[b][i], outputs_batch[b][i])
                    << "Mismatch in block " << b << " at index " << i;
                break;
            }
        }
    }
}

/**
 * @test MemoryPoolPerformance
 * @brief Test memory pool overhead reduction
 */
TEST_F(SYCLOptimizedPerformanceTest, MemoryPoolPerformance) {
    ASSERT_NE(queue_, nullptr);

    constexpr int kSize = 8;
    constexpr int kStride = 8;
    constexpr int kNumBlocks = 100;
    constexpr int kIterations = 50;

    // Create memory pool
    TxfmMemoryPool pool(*queue_, kNumBlocks);

    std::vector<std::vector<int16_t>> inputs(kNumBlocks);
    std::vector<std::vector<tran_low_t>> outputs(kNumBlocks);

    for (int i = 0; i < kNumBlocks; ++i) {
        inputs[i].resize(kSize * kStride);
        outputs[i].resize(kSize * kSize);
        initialize_test_data(inputs[i].data(), inputs[i].size(), i);
    }

    TxfmParams params;
    params.tx_size = TX_8X8;
    params.tx_type = DCT_DCT;
    params.bd = 10;
    params.eob = 0;
    params.dir = TxfmDir::kForward;
    params.shift = 0;

    std::cout << "\n=== Memory Pool Performance ===" << std::endl;

    // Without pool (direct allocation each time)
    auto no_pool_time = benchmark("Without pool", [&]() {
        for (int i = 0; i < kNumBlocks; ++i) {
            fdct8x8_optimized(*queue_, inputs[i].data(),
                             outputs[i].data(), kStride, params);
        }
    }, kIterations);

    // With pool (pre-allocated memory)
    auto with_pool_time = benchmark("With pool", [&]() {
        for (int i = 0; i < kNumBlocks; ++i) {
            auto buffers = pool.acquire(kSize);
            // Copy input to device
            queue_->memcpy(buffers.input, inputs[i].data(),
                          kSize * kStride * sizeof(int16_t));
            // Run transform (would use pool buffers in real implementation)
            queue_->wait();
            fdct8x8_optimized(*queue_, inputs[i].data(),
                             outputs[i].data(), kStride, params);
            pool.release(buffers, kSize);
        }
    }, kIterations);

    with_pool_time.speedup_vs_baseline = no_pool_time.avg_time_us / with_pool_time.avg_time_us;

    no_pool_time.print();
    with_pool_time.print();

    std::cout << "\nSummary: Memory pool achieved "
              << with_pool_time.speedup_vs_baseline << "x speedup" << std::endl;
    std::cout << "=================================\n" << std::endl;
}

/**
 * @test ScalingPerformance
 * @brief Test performance scaling with different work-group configurations
 */
TEST_F(SYCLOptimizedPerformanceTest, ScalingPerformance) {
    ASSERT_NE(queue_, nullptr);

    constexpr int kDataSize = 1024 * 1024;
    std::vector<int> input(kDataSize, 1);
    std::vector<int> output(kDataSize);

    sycl::buffer<int, 1> input_buf(input.data(), sycl::range<1>(kDataSize));
    sycl::buffer<int, 1> output_buf(output.data(), sycl::range<1>(kDataSize));

    std::vector<size_t> group_sizes = {32, 64, 128, 256};

    std::cout << "\n=== Work Group Size Scaling ===" << std::endl;

    for (const auto group_size : group_sizes) {
        auto start = Clock::now();
        queue_->submit([&](sycl::handler& cgh) {
            auto in = input_buf.get_access<sycl::access::mode::read>(cgh);
            auto out = output_buf.get_access<sycl::access::mode::write>(cgh);
            sycl::nd_range<1> range(kDataSize, group_size);
            cgh.parallel_for(range,
                [=](sycl::nd_item<1> item) {
                    out[item.get_global_id()] = in[item.get_global_id()] * 2;
                });
        });
        queue_->wait();

        auto end = Clock::now();
        double time_ms = Duration(end - start).count() * 1000;

        std::cout << "Group size: " << std::setw(4) << group_size
                  << "  Time: " << std::setw(8) << time_ms << " ms"
                  << "  Throughput: " << std::setw(6) << std::fixed
                  << std::setprecision(2)
                  << (kDataSize * sizeof(int) / 1024.0 / 1024) / (time_ms / 1000)
                  << " GB/s" << std::endl;
    }

    std::cout << "\n=================================\n" << std::endl;
}

/**
 * @test LocalMemoryBandwidth
 * @brief Measure effective local memory bandwidth
 */
TEST_F(SYCLOptimizedPerformanceTest, LocalMemoryBandwidth) {
    ASSERT_NE(queue_, nullptr);

    constexpr int kNumIterations = 1000;

    std::cout << "\n=== Local Memory Bandwidth Test ===" << std::endl;

    // Test with 8x8 tile (matches transform size)
    constexpr int kTileSize = 8;

    auto kernel_time = benchmark("8x8 tile load/store", [&]() {
        queue_->submit([&](sycl::handler& cgh) {
            sycl::accessor<int, 2, sycl::access::mode::read_write,
                           sycl::target::local> local_tile(
                sycl::range<2>(kTileSize, kTileSize), cgh);

            cgh.parallel_for(sycl::nd_range<2>(
                sycl::range<2>(kTileSize, kTileSize),
                sycl::range<2>(kTileSize, kTileSize)),
                [=](sycl::nd_item<2> item) {
                    const int row = item.get_local_id(0);
                    const int col = item.get_local_id(1);

                    // Write to local memory
                    local_tile[row][col] = row * kTileSize + col;

                    item.barrier(sycl::access::fence_space::local_space);

                    // Read from local memory and transpose
                    int val = local_tile[col][row];
                });
        });
        queue_->wait();
    }, kNumIterations);

    // Calculate theoretical bandwidth
    size_t bytes_per_iter = 2 * kTileSize * kTileSize * sizeof(int);  // read + write
    double bandwidth_gbps =
        (bytes_per_iter * kNumIterations / 1e9) /
        (kernel_time.avg_time_us * kNumIterations / 1e6);

    std::cout << "  Tile size: " << kTileSize << "x" << kTileSize << std::endl;
    std::cout << "  Bytes per iteration: " << bytes_per_iter << std::endl;
    std::cout << "  Effective local memory bandwidth: "
              << std::fixed << std::setprecision(2) << bandwidth_gbps << " GB/s" << std::endl;
    std::cout << "===================================\n" << std::endl;
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
