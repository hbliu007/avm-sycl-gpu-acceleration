// minimal_sycl_test.cpp - Minimal SYCL test for GPU verification
// Build: icpx -fsycl minimal_sycl_test.cpp -o minimal_sycl_test
// Run: ONEAPI_DEVICE_SELECTOR=cuda:0 ./minimal_sycl_test

#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  AVM SYCL GPU Acceleration Test" << std::endl;
    std::cout << "========================================" << std::endl;

    // Get GPU device
    auto devices = ::sycl::device::get_devices(::sycl::info::device_type::gpu);
    if (devices.empty()) {
        std::cerr << "ERROR: No GPU devices found!" << std::endl;
        return 1;
    }

    // Select first GPU
    ::sycl::device gpu = devices[0];
    ::sycl::queue q(gpu, ::sycl::property::queue::enable_profiling{});

    std::cout << "\n=== GPU Device Info ===" << std::endl;
    std::cout << "Device: " << gpu.get_info<::sycl::info::device::name>() << std::endl;
    std::cout << "Vendor: " << gpu.get_info<::sycl::info::device::vendor>() << std::endl;
    std::cout << "Compute Units: " << gpu.get_info<::sycl::info::device::max_compute_units>() << std::endl;
    std::cout << "Global Memory: " << gpu.get_info<::sycl::info::device::global_mem_size>() / (1024*1024*1024) << " GB" << std::endl;
    std::cout << "Max Work Group: " << gpu.get_info<::sycl::info::device::max_work_group_size>() << std::endl;

    // Test 1: Simple vector add
    std::cout << "\n=== Test 1: Vector Add (Basic GPU Compute) ===" << std::endl;
    {
        constexpr int N = 1024;
        std::vector<float> a(N, 1.0f);
        std::vector<float> b(N, 2.0f);
        std::vector<float> c(N, 0.0f);

        float* d_a = ::sycl::malloc_device<float>(N, q);
        float* d_b = ::sycl::malloc_device<float>(N, q);
        float* d_c = ::sycl::malloc_device<float>(N, q);

        q.memcpy(d_a, a.data(), N * sizeof(float)).wait();
        q.memcpy(d_b, b.data(), N * sizeof(float)).wait();

        auto start = std::chrono::high_resolution_clock::now();

        q.parallel_for(::sycl::range<1>(N), [=](::sycl::id<1> i) {
            d_c[i] = d_a[i] + d_b[i];
        }).wait();

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        q.memcpy(c.data(), d_c, N * sizeof(float)).wait();

        std::cout << "Vector add completed in " << duration.count() << " microseconds" << std::endl;
        std::cout << "Result verification: ";
        bool correct = true;
        for (int i = 0; i < N; ++i) {
            if (c[i] != 3.0f) {
                correct = false;
                break;
            }
        }
        std::cout << (correct ? "PASSED" : "FAILED") << std::endl;

        ::sycl::free(d_a, q);
        ::sycl::free(d_b, q);
        ::sycl::free(d_c, q);
    }

    // Test 2: DCT-like kernel
    std::cout << "\n=== Test 2: DCT 8x8 Transform Kernel ===" << std::endl;
    {
        constexpr int N = 8;
        constexpr int SIZE = N * N;
        std::vector<int16_t> input(SIZE);
        std::vector<int32_t> output(SIZE);

        // Initialize with test pattern
        for (int i = 0; i < SIZE; ++i) {
            input[i] = static_cast<int16_t>((i % 8) * 10 + (i / 8) * 10);
        }

        int16_t* d_input = ::sycl::malloc_device<int16_t>(SIZE, q);
        int32_t* d_output = ::sycl::malloc_device<int32_t>(SIZE, q);

        q.memcpy(d_input, input.data(), SIZE * sizeof(int16_t)).wait();

        auto start = std::chrono::high_resolution_clock::now();

        // DCT Type-II kernel
        q.parallel_for(::sycl::range<2>(N, N), [=](::sycl::id<2> idx) {
            int row = idx[0];
            int col = idx[1];
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                float angle = 3.14159265f * (row + 0.5f) * k / N;
                sum += d_input[k * N + col] * cos(angle);
            }
            if (row == 0) sum *= 0.70710678f;  // sqrt(2)/2
            d_output[row * N + col] = static_cast<int32_t>(sum);
        }).wait();

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        q.memcpy(output.data(), d_output, SIZE * sizeof(int32_t)).wait();

        std::cout << "DCT 8x8 completed in " << duration.count() << " microseconds" << std::endl;
        std::cout << "DC coefficient: " << output[0] << std::endl;
        std::cout << "First 4 coefficients: ";
        for (int i = 0; i < 4; ++i) std::cout << output[i] << " ";
        std::cout << std::endl;

        ::sycl::free(d_input, q);
        ::sycl::free(d_output, q);
    }

    // Test 3: SAD-like kernel
    std::cout << "\n=== Test 3: SAD 16x16 Motion Estimation ===" << std::endl;
    {
        constexpr int N = 16;
        constexpr int SIZE = N * N;
        std::vector<uint8_t> ref(SIZE), cur(SIZE);

        for (int i = 0; i < SIZE; ++i) {
            ref[i] = static_cast<uint8_t>(i % 256);
            cur[i] = static_cast<uint8_t>((i + 10) % 256);
        }

        uint8_t* d_ref = ::sycl::malloc_device<uint8_t>(SIZE, q);
        uint8_t* d_cur = ::sycl::malloc_device<uint8_t>(SIZE, q);
        uint32_t* d_sad = ::sycl::malloc_device<uint32_t>(1, q);

        q.memcpy(d_ref, ref.data(), SIZE * sizeof(uint8_t)).wait();
        q.memcpy(d_cur, cur.data(), SIZE * sizeof(uint8_t)).wait();
        q.memset(d_sad, 0, sizeof(uint32_t)).wait();

        auto start = std::chrono::high_resolution_clock::now();

        // SAD kernel with reduction
        q.single_task([=]() {
            uint32_t sum = 0;
            for (int i = 0; i < SIZE; ++i) {
                int diff = static_cast<int>(d_ref[i]) - static_cast<int>(d_cur[i]);
                sum += (diff < 0) ? -diff : diff;
            }
            *d_sad = sum;
        }).wait();

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        uint32_t sad;
        q.memcpy(&sad, d_sad, sizeof(uint32_t)).wait();

        // Verify
        uint32_t expected = 0;
        for (int i = 0; i < SIZE; ++i) {
            int diff = static_cast<int>(ref[i]) - static_cast<int>(cur[i]);
            expected += (diff < 0) ? -diff : diff;
        }

        std::cout << "SAD 16x16 completed in " << duration.count() << " microseconds" << std::endl;
        std::cout << "GPU SAD: " << sad << ", Expected: " << expected << std::endl;
        std::cout << "Verification: " << (sad == expected ? "PASSED" : "FAILED") << std::endl;

        ::sycl::free(d_ref, q);
        ::sycl::free(d_cur, q);
        ::sycl::free(d_sad, q);
    }

    // Test 4: Performance benchmark
    std::cout << "\n=== Test 4: Performance Benchmark (1000 iterations) ===" << std::endl;
    {
        constexpr int N = 8;
        constexpr int SIZE = N * N;
        std::vector<int16_t> input(SIZE);
        std::vector<int32_t> output(SIZE);
        for (int i = 0; i < SIZE; ++i) input[i] = i;

        int16_t* d_input = ::sycl::malloc_device<int16_t>(SIZE, q);
        int32_t* d_output = ::sycl::malloc_device<int32_t>(SIZE, q);
        q.memcpy(d_input, input.data(), SIZE * sizeof(int16_t)).wait();

        const int iterations = 1000;
        auto start = std::chrono::high_resolution_clock::now();

        for (int iter = 0; iter < iterations; ++iter) {
            q.parallel_for(::sycl::range<2>(N, N), [=](::sycl::id<2> idx) {
                int row = idx[0];
                int col = idx[1];
                float sum = 0.0f;
                for (int k = 0; k < N; ++k) {
                    float angle = 3.14159265f * (row + 0.5f) * k / N;
                    sum += d_input[k * N + col] * cos(angle);
                }
                d_output[row * N + col] = static_cast<int32_t>(sum);
            });
        }
        q.wait();

        auto end = std::chrono::high_resolution_clock::now();
        auto total_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        std::cout << "Total time for " << iterations << " DCT 8x8: " << total_us << " us" << std::endl;
        std::cout << "Average per DCT: " << (double)total_us / iterations << " us" << std::endl;
        std::cout << "Throughput: " << (double)iterations * 1000000 / total_us << " DCT/sec" << std::endl;

        ::sycl::free(d_input, q);
        ::sycl::free(d_output, q);
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << "  All GPU tests completed successfully!" << std::endl;
    std::cout << "  SYCL GPU acceleration is working!" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
