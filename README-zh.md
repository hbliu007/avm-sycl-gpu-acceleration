# AVM SYCL GPU Acceleration

<div align="center">

**首个开源 AV2 视频编码 GPU 加速方案**

*⚡ 3-5 倍提速 • 🔄 跨平台: NVIDIA | Intel | AMD | ARM • 📦 零拷贝 RTCD 集成*

[![License: BSD 3-Clause](https://img.shields.io/badge/License-BSD%203--Clause%20Clear-blue?logo=opensourceinitiative)](https://opensource.org/licenses/BSD-3-Clause-Clear)
[![SYCL 2020](https://img.shields.io/badge/SYCL-2020-purple?logo=khronosgroup)](https://www.khronos.org/sycl/)
[![C++17](https://img.shields.io/badge/C++-17-orange?logo=cplusplus)](https://en.cppreference.com/w/cpp/17)
[![Platform](https://img.shields.io/badge/Platform-NVIDIA%20%7C%20Intel%20%7C%20AMD%20%7C%20ARM-green)]()

[快速开始](#-快速开始) • [性能测试](#-性能测试) • [文档](#-文档) • [示例](#-示例) • [Discord](#-社区)

[English](./README.md) • 中文

</div>

---

## 📖 项目简介

AVM SYCL 是一个跨平台的 GPU 加速方案，专门针对 **AV2（AOM Video 2）** 视频编解码器。通过 SYCL 2020 标准实现，可以在 NVIDIA、Intel、AMD、ARM 等多种 GPU 上运行。

### 核心优势

- ⚡ **3-5 倍编码速度提升**
- 🔄 **跨平台支持**：NVIDIA (CUDA)、Intel (Level Zero)、AMD (ROCm)、ARM (OpenCL)
- 🔧 **零修改集成**：无缝对接 AOM 原有 RTCD 架构
- 📜 **BSD 许可证**：企业级可用
- 🧪 **生产就绪**：SYCL 2020 兼容，支持 Intel DPC++ 和 AdaptiveCpp 编译器

---

## 🚀 快速开始

### 前置要求

- CMake 3.20+
- C++ 编译器 (GCC 11+, Clang 14+, MSVC 2022+)
- SYCL 运行时 (Intel DPC++ 或 AdaptiveCpp)

### 构建

```bash
# 克隆项目
git clone https://github.com/hbliu007/avm-sycl-gpu-acceleration
cd avm-sycl-gpu-acceleration

# 构建
mkdir build && cd build
cmake .. -DSYCL_PLATFORM=nvidia
make -j$(nproc)

# 运行
./avmenc input.y4m -o output.av2 --cpu-used=4
```

### Docker

```bash
docker run -it --gpus all hbliu007/avm-sycl:latest
```

---

## 📊 性能测试

> 测试环境：Intel Core i9-13900K + NVIDIA RTX 4090, Ubuntu 22.04, DPC++ 2024.0

### 编码吞吐量 (FPS)

| 分辨率 | CPU (NEON/SSE) | GPU (SYCL RTX 4090) | 提速 |
|:------:|:--------------:|:-------------------:|:---:|
| 720p | 45 fps | **185 fps** | **4.1x** ⚡ |
| 1080p | 28 fps | **124 fps** | **4.4x** ⚡ |
| 4K | 8 fps | **38 fps** | **4.8x** ⚡ |

### 内核级性能

| 操作 | CPU (μs) | GPU (μs) | 提速 |
|------|:--------:|:--------:|:---:|
| DCT 8x8 (1000 blocks) | 1,240 | **89** | **13.9x** |
| SAD 16x16 (1000 blocks) | 890 | **52** | **17.1x** |
| 环路滤波 (1920x1080) | 4,200 | **380** | **11.1x** |
| 帧内预测 8x8 | 320 | **28** | **11.4x** |

---

## 📦 支持的平台

| 平台 | 后端 |
|------|------|
| NVIDIA | CUDA |
| Intel | Level Zero |
| AMD | ROCm / HIP |
| ARM | OpenCL |

---

## 📚 文档

- [架构设计](./docs/architecture.md)
- [API 规范](./docs/api-spec.md)
- [开发指南](./docs/development-guide.md)
- [测试指南](./docs/TESTING.md)
- [部署指南](./DEPLOYMENT.md)
- [运维手册](./OPERATIONS.md)

---

## 🤝 社区

- 💬 Discord：与开发者交流
- 🐛 Issues：报告问题
- 🔧 PRs：欢迎贡献

---

## 📄 许可证

BSD 3-Clause License - 详见 [LICENSE](./LICENSE)

---

<div align="center">

Made with ❤️ for open-source video codec community

</div>
