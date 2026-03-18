// avm_dsp/sycl/sycl_context.hpp
/*
 * Copyright (c) 2026, Alliance for Open Media. All rights reserved.
 *
 * SYCL GPU context management for AV2 codec.
 */

#ifndef AVM_DSP_SYCL_SYCL_CONTEXT_HPP_
#define AVM_DSP_SYCL_SYCL_CONTEXT_HPP_

#ifdef HAVE_SYCL

#include <sycl/sycl.hpp>
#include <string>
#include <vector>

namespace avm {
namespace sycl {

// Platform information structure
struct PlatformInfo {
    std::string name;
    std::string vendor;
    bool is_gpu;
    bool is_cpu;
    size_t compute_units;
    size_t global_mem_size;
};

// Singleton SYCL context manager
class SYCLContext {
public:
    // Get singleton instance
    static SYCLContext& instance();

    // Initialize context (called automatically on first use)
    bool initialize();

    // Check if SYCL is available and initialized
    bool is_available() const { return available_; }

    // Check if using GPU device
    bool is_gpu() const { return is_gpu_; }

    // Get SYCL queue
    ::sycl::queue& queue() { return queue_; }

    // Get current device
    ::sycl::device get_device() { return device_; }

    // Get compute unit count
    size_t compute_units() const { return compute_units_; }

    // Get global memory size
    size_t global_mem_size() const { return global_mem_size_; }

    // List all available devices
    std::vector<PlatformInfo> list_devices();

    // Get backend name
    std::string backend_name() const { return backend_name_; }

private:
    SYCLContext();
    ~SYCLContext() = default;

    // Prevent copying
    SYCLContext(const SYCLContext&) = delete;
    SYCLContext& operator=(const SYCLContext&) = delete;

    // Score device for selection priority
    int score_device(const ::sycl::device& dev);

    // Select best available device
    ::sycl::device select_best_device();

    ::sycl::device device_;
    ::sycl::queue queue_;

    bool available_ = false;
    bool is_gpu_ = false;
    bool initialized_ = false;

    size_t compute_units_ = 0;
    size_t global_mem_size_ = 0;

    std::string backend_name_;
};

}  // namespace sycl
}  // namespace avm

#endif  // HAVE_SYCL

#endif  // AVM_DSP_SYCL_SYCL_CONTEXT_HPP_
