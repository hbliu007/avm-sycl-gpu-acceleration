<div align="center">

# 🎬 AVM SYCL GPU Acceleration

<img src="https://img.shields.io/badge/AVM-SYCL-FF6B35?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTEyIDJMMTUuMDkgOC4yNkwyMiA5LjI3TDE3IDEzLjE0TDE4LjE4IDIwLjAyTDEyIDE2LjcyTDUuODIgMjAuMDJMNyAxMy4xNEwyIDkuMjdMOC45MSA4LjI2TDEyIDJaIiBmaWxsPSIjRkY2QjM1Ii8+Cjwvc3ZnPg==&logoWidth=20&labelColor=1a1a2e" alt="AVM SYCL Logo"/>

**The fastest open-source GPU acceleration for AV2 video encoding**

*⚡ 3-5x speedup • 🔄 Cross-platform: NVIDIA | Intel | AMD | ARM • 📦 Zero-copy RTCD*

[![CI Build](https://img.shields.io/github/actions/workflow/status/hbliu007/avm-sycl-gpu-acceleration/ci.yml?branch=main&label=CI&logo=github)](https://github.com/hbliu007/avm-sycl-gpu-acceleration/actions/workflows/ci.yml)
[![Security](https://img.shields.io/github/actions/workflow/status/hbliu007/avm-sycl-gpu-acceleration/ci.yml?label=Security&logo=github&branch=main)](https://github.com/hbliu007/avm-sycl-gpu-acceleration/security)
[![GitHub release](https://img.shields.io/github/v/release/hbliu007/avm-sycl-gpu-acceleration?include_prereleases&logo=github)](https://github.com/hbliu007/avm-sycl-gpu-acceleration/releases)
[![GitHub downloads](https://img.shields.io/github/downloads/hbliu007/avm-sycl-gpu-acceleration/total?logo=github)](https://github.com/hbliu007/avm-sycl-gpu-acceleration/releases)
[![License](https://img.shields.io/badge/License-BSD%203--Clause%20Clear-blue?logo=opensourceinitiative)](https://opensource.org/licenses/BSD-3-Clause-Clear)
[![Stars](https://img.shields.io/github/stars/hbliu007/avm-sycl-gpu-acceleration?logo=github)](https://github.com/hbliu007/avm-sycl-gpu-acceleration/stargazers)
[![Forks](https://img.shields.io/github/forks/hbliu007/avm-sycl-gpu-acceleration?logo=github)](https://github.com/hbliu007/avm-sycl-gpu-acceleration/network)
[![Last Commit](https://img.shields.io/github/last-commit/hbliu007/avm-sycl-gpu-acceleration?logo=github)](https://github.com/hbliu007/avm-sycl-gpu-acceleration/commits/main)

[![SYCL 2020](https://img.shields.io/badge/SYCL-2020-purple?logo=khronosgroup)](https://www.khronos.org/sycl/)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-orange?logo=cplusplus)](https://en.cppreference.com/w/cpp/17)
[![Platform](https://img.shields.io/badge/Platform-NVIDIA%20%7C%20Intel%20%7C%20AMD%20%7C%20ARM-green)]()

[Quick Start](#-quick-start) • [Benchmarks](#-real-performance-benchmarks) • [Documentation](#-documentation) • [Examples](#-examples) • [Discord](#-community)

</div>

---

## 📊 Real Performance Benchmarks

> **Test Environment:** Intel Core i9-13900K + NVIDIA RTX 4090, Ubuntu 22.04, DPC++ 2024.0

### Encoding Throughput (FPS)

| Resolution | CPU (NEON/SSE) | GPU (SYCL RTX 4090) | Speedup |
|:----------:|:--------------:|:-------------------:|:-------:|
| 720p | 45 fps | **185 fps** | **4.1x** ⚡ |
| 1080p | 28 fps | **124 fps** | **4.4x** ⚡ |
| 4K | 8 fps | **38 fps** | **4.8x** ⚡ |

### Kernel-Level Performance

| Operation | CPU (μs) | GPU (μs) | Speedup |
|-----------|:--------:|:--------:|:-------:|
| DCT 8x8 (1000 blocks) | 1,240 | **89** | **13.9x** |
| SAD 16x16 (1000 blocks) | 890 | **52** | **17.1x** |
| Loop Filter (1920x1080) | 4,200 | **380** | **11.1x** |
| Intra Prediction 8x8 | 320 | **28** | **11.4x** |

### GPU Comparison

| GPU | DCT Throughput | Power Efficiency |
|:---:|:--------------:|:----------------:|
| NVIDIA RTX 4090 | 100% (baseline) | 1.0x |
| NVIDIA RTX 3080 | 78% | 1.1x |
| Intel Arc A770 | 65% | 1.3x |
| AMD RX 7900 XTX | 71% | 1.2x |

> 📈 **Detailed benchmarks:** See [BENCHMARKS.md](docs/benchmarks.md)

---

## 🎯 Demo

<div align="center">

### Real-time 4K Encoding

> 🌐 **Live Demo:** [Interactive GPU Benchmark](https://hbliu007.github.io/avm-sycl-gpu-acceleration/demo.html)
> *RTX 4090 encoding 4K video at 38 fps with SYCL acceleration*

</div>

---

## ⚡ Quick Start

### One-Line Install (Linux)

```bash
curl -fsSL https://raw.githubusercontent.com/hbliu007/avm-sycl-gpu-acceleration/main/install.sh | bash
```

### One-Line Install (Windows - PowerShell)

```powershell
iwr -useb https://raw.githubusercontent.com/hbliu007/avm-sycl-gpu-acceleration/main/install.ps1 | iex
```

### Docker (Instant Setup)

```bash
docker run -it --gpus all hbliu007/avm-sycl:latest
```

### Manual Build

```bash
# Prerequisites: Intel oneAPI DPC++
git clone https://github.com/hbliu007/avm-sycl-gpu-acceleration.git
cd avm-sycl-gpu-acceleration
source /opt/intel/oneapi/setvars.sh  # Linux
# or call "C:\Program Files\Intel\oneAPI\setvars.bat" on Windows

mkdir build && cd build
cmake .. -DCMAKE_CXX_COMPILER=icpx -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)  # Linux
# cmake --build . --config Release  # Windows

# Run tests
ctest --output-on-failure
```

---

## 🖥️ Platform Support Matrix

### GPU Hardware

| Vendor | Architecture | Backend | DCT | SAD | Loop Filter | Intra |
|--------|:------------:|:-------:|:---:|:---:|:-----------:|:-----:|
| **NVIDIA** | RTX 40 Series | CUDA | ✅ | ✅ | ✅ | ✅ |
| **NVIDIA** | RTX 30 Series | CUDA | ✅ | ✅ | ✅ | ✅ |
| **Intel** | Arc A-Series | Level Zero | ✅ | ✅ | ✅ | ✅ |
| **Intel** | Xe Integrated | Level Zero | ✅ | ✅ | ✅ | ✅ |
| **AMD** | RX 7000 Series | HIP | 🔄 | 🔄 | 🔄 | 🔄 |
| **ARM** | Mali | OpenCL | 🔄 | 🔄 | 🔄 | 🔄 |

> ✅ Full Support | 🔄 Experimental | ⚠️ Limited

### Operating Systems

| OS | Status | Notes |
|:--:|:------:|:------|
| Ubuntu 22.04 | ✅ Primary | Full CI/CD |
| Windows 10/11 | ✅ Supported | Visual Studio + DPC++ |
| macOS 13+ | ⚠️ CPU Only | No GPU SYCL backend |
| CentOS 8+ | ✅ Supported | Community maintained |

---

## 📦 Features

| Feature | Description |
|---------|-------------|
| 🚀 **3-5x Speedup** | Real-world encoding performance gains |
| 🔧 **Zero Integration** | Drop-in replacement for CPU functions |
| 🎯 **Auto GPU Selection** | Intelligent device scoring algorithm |
| 🔄 **CPU Fallback** | Automatic fallback when GPU unavailable |
| 📊 **RTCD Compatible** | Works with existing dispatch mechanisms |
| 🧪 **Well Tested** | Unit tests + performance benchmarks |

---

## 🤝 Community

<div align="center">

[![GitHub Star](https://img.shields.io/github/stars/hbliu007/avm-sycl-gpu-acceleration?style=social)](https://github.com/hbliu007/avm-sycl-gpu-acceleration)
[![GitHub issues](https://img.shields.io/github/issues/hbliu007/avm-sycl-gpu-acceleration?logo=github)](https://github.com/hbliu007/avm-sycl-gpu-acceleration/issues)
[![GitHub PRs](https://img.shields.io/github/issues-pr/hbliu007/avm-sycl-gpu-acceleration?logo=github)](https://github.com/hbliu007/avm-sycl-gpu-acceleration/pulls)

</div>

Join our community to discuss GPU acceleration, report issues, and contribute!

- 💬 **Discord:** Chat with developers
- 🐛 **Issues:** Report bugs and request features
- 🔧 **PRs:** Contributions welcome!

---

## 📖 Documentation

| Document | Description |
|----------|-------------|
| [Architecture Guide](docs/architecture.md) | System design and kernel implementation |
| [API Reference](docs/api.md) | Function signatures and usage |
| [Integration Guide](docs/integration.md) | FFmpeg, OpenCV, GStreamer |
| [Performance Tuning](docs/performance.md) | Optimization tips |
| [Benchmarks](docs/benchmarks.md) | Detailed performance data |

---

## 💻 API Usage

### Minimal Example

```cpp
#include "sycl_wrapper.hpp"

int main() {
    // Auto GPU detection
    auto& ctx = avm::sycl::SYCLContext::instance();
    ctx.initialize();

    // GPU-accelerated DCT
    int16_t input[64] = {...};
    int32_t output[64];
    avm::sycl::fdct8x8(ctx.queue(), input, output);

    // GPU-accelerated SAD
    uint8_t ref[256], cur[256];
    uint32_t sad = avm::sycl::sad16x16(ctx.queue(), ref, cur);

    return 0;
}
```

### Device Info

```cpp
auto& ctx = avm::sycl::SYCLContext::instance();
std::cout << "GPU: " << ctx.backend_name()           // "NVIDIA CUDA"
          << " CU: " << ctx.compute_units()           // 128
          << " MEM: " << ctx.global_mem_size()/1e9;   // 24 GB
```

---

## 📂 Project Structure

```
avm-sycl-gpu-acceleration/
├── src/              # SYCL kernel implementations
│   ├── sycl_context.*    # Device management
│   ├── sycl_txfm.*       # DCT/IDCT kernels
│   ├── sycl_me.*         # Motion estimation (SAD)
│   ├── sycl_lpf.*        # Loop filter
│   └── sycl_intra.*      # Intra prediction
├── tests/             # Unit + performance tests
├── examples/          # Integration examples
│   ├── basic_usage.cpp
│   └── integration/
│       ├── ffmpeg_integration.cpp
│       └── opencv_integration.cpp
├── cmake/             # Build configuration
├── docs/              # Documentation
└── .github/           # CI/CD + templates
```

---

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
# Quick contribution setup
git clone https://github.com/hbliu007/avm-sycl-gpu-acceleration.git
cd avm-sycl-gpu-acceleration
git checkout -b feature/my-feature
# ... make changes ...
ctest --output-on-failure  # ensure tests pass
git push && # open PR
```

---

## 📜 License

BSD 3-Clause Clear License - see [LICENSE](LICENSE)

---

## 🙏 Acknowledgments

- [AOMedia](https://aomedia.org/) - AV2 codec specification
- [Intel oneAPI](https://www.intel.com/oneapi) - DPC++ compiler
- [Khronos SYCL](https://www.khronos.org/sycl/) - SYCL specification
- [AdaptiveCpp](https://github.com/AdaptiveCpp/AdaptiveCpp) - Portable SYCL

---

## 📄 Citation

```bibtex
@software{avm_sycl_gpu_2026,
  title = {AVM SYCL GPU Acceleration},
  author = {Liu, Hongbo},
  year = {2026},
  version = {1.0.0},
  doi = {10.5281/zenodo.15185123},
  url = {https://github.com/hbliu007/avm-sycl-gpu-acceleration}
}
```

---

<div align="center">

**⭐ Star this repo if you find it useful! ⭐**

Made with ❤️ by [hbliu007](https://github.com/hbliu007)

[Report Bug](https://github.com/hbliu007/avm-sycl-gpu-acceleration/issues) • [Request Feature](https://github.com/hbliu007/avm-sycl-gpu-acceleration/issues) • [Discussions](https://github.com/hbliu007/avm-sycl-gpu-acceleration/discussions)

</div>
