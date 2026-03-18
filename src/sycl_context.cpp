// avm_dsp/sycl/sycl_context.cpp
/*
 * Copyright (c) 2026, Alliance for Open Media. All rights reserved.
 *
 * SYCL GPU context management implementation.
 */

#ifdef HAVE_SYCL

#include "sycl_context.hpp"
#include <algorithm>
#include <iostream>

namespace avm {
namespace sycl {

SYCLContext& SYCLContext::instance() {
    static SYCLContext ctx;
    return ctx;
}

SYCLContext::SYCLContext() {
    initialize();
}

bool SYCLContext::initialize() {
    if (initialized_) {
        return available_;
    }
    initialized_ = true;

    try {
        // Get all available devices
        auto devices = ::sycl::device::get_devices(::sycl::info::device_type::all);

        if (devices.empty()) {
            std::cerr << "[SYCL] No devices found" << std::endl;
            return false;
        }

        // Select best device
        device_ = select_best_device();

        if (!device_.is_gpu() && !device_.is_cpu()) {
            std::cerr << "[SYCL] No suitable device found" << std::endl;
            return false;
        }

        // Create queue with profiling enabled
        queue_ = ::sycl::queue(device_,
            ::sycl::property::queue::enable_profiling{});

        // Store device properties
        is_gpu_ = device_.is_gpu();
        compute_units_ = device_.get_info<::sycl::info::device::max_compute_units>();
        global_mem_size_ = device_.get_info<::sycl::info::device::global_mem_size>();

        // Get backend name
        auto platform = device_.get_platform();
        backend_name_ = platform.get_info<::sycl::info::platform::name>();

        available_ = true;

        std::cout << "[SYCL] Initialized successfully" << std::endl;
        std::cout << "  Device: " << device_.get_info<::sycl::info::device::name>() << std::endl;
        std::cout << "  Backend: " << backend_name_ << std::endl;
        std::cout << "  Type: " << (is_gpu_ ? "GPU" : "CPU") << std::endl;
        std::cout << "  Compute units: " << compute_units_ << std::endl;
        std::cout << "  Global memory: " << (global_mem_size_ / 1024 / 1024) << " MB" << std::endl;

        return true;

    } catch (const ::sycl::exception& e) {
        std::cerr << "[SYCL] Initialization failed: " << e.what() << std::endl;
        available_ = false;
        return false;
    }
}

int SYCLContext::score_device(const ::sycl::device& dev) {
    int score = 0;

    // GPU is strongly preferred
    if (dev.is_gpu()) {
        score += 1000;
    }

    // Vendor scoring
    std::string vendor = dev.get_info<::sycl::info::device::vendor>();
    std::transform(vendor.begin(), vendor.end(), vendor.begin(), ::tolower);

    if (vendor.find("nvidia") != std::string::npos) {
        score += 300;
    } else if (vendor.find("intel") != std::string::npos) {
        score += 200;
    } else if (vendor.find("amd") != std::string::npos) {
        score += 150;
    } else if (vendor.find("apple") != std::string::npos) {
        score += 100;
    } else if (vendor.find("arm") != std::string::npos) {
        score += 100;
    }

    // Compute units
    score += dev.get_info<::sycl::info::device::max_compute_units>();

    // Memory bonus
    size_t mem = dev.get_info<::sycl::info::device::global_mem_size>();
    score += static_cast<int>(mem / (1024 * 1024 * 100));  // 1 point per 100MB

    return score;
}

::sycl::device SYCLContext::select_best_device() {
    auto devices = ::sycl::device::get_devices(::sycl::info::device_type::all);

    ::sycl::device best = devices[0];
    int best_score = score_device(best);

    for (auto& dev : devices) {
        int score = score_device(dev);
        if (score > best_score) {
            best_score = score;
            best = dev;
        }
    }

    return best;
}

std::vector<PlatformInfo> SYCLContext::list_devices() {
    std::vector<PlatformInfo> infos;

    auto devices = ::sycl::device::get_devices(::sycl::info::device_type::all);

    for (auto& dev : devices) {
        PlatformInfo info;
        info.name = dev.get_info<::sycl::info::device::name>();
        info.vendor = dev.get_info<::sycl::info::device::vendor>();
        info.is_gpu = dev.is_gpu();
        info.is_cpu = dev.is_cpu();
        info.compute_units = dev.get_info<::sycl::info::device::max_compute_units>();
        info.global_mem_size = dev.get_info<::sycl::info::device::global_mem_size>();
        infos.push_back(info);
    }

    return infos;
}

}  // namespace sycl
}  // namespace avm

#endif  // HAVE_SYCL
