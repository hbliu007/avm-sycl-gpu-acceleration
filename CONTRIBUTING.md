# Contributing to AVM SYCL GPU Acceleration

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing.

## Getting Started

### Prerequisites

- C++17 compatible compiler
- Intel DPC++ (icpx) or AdaptiveCpp (hipSYCL)
- CMake 3.20+
- Git

### Development Setup

```bash
# Fork and clone
git clone https://github.com/hbliu007/avm-sycl-gpu-acceleration.git
cd avm-sycl-gpu-acceleration

# Create feature branch
git checkout -b feature/your-feature-name

# Build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug -DENABLE_TESTS=ON
make -j$(nproc)

# Run tests
ctest --output-on-failure
```

## Contribution Types

### Bug Reports

When reporting bugs, please include:

1. **Environment**: OS, compiler, GPU model
2. **Steps to reproduce**: Minimal code example
3. **Expected behavior**: What should happen
4. **Actual behavior**: What actually happens
5. **Logs**: Build output or runtime logs

### Feature Requests

For new features:

1. **Use case**: Why is this feature needed?
2. **Proposed API**: Suggested interface
3. **Implementation ideas**: Optional technical approach

### Code Contributions

#### Code Style

- Follow C++ Core Guidelines
- Use 4-space indentation
- Maximum line length: 100 characters
- Use `snake_case` for functions/variables
- Use `PascalCase` for types/classes

#### Commit Messages

```
type(scope): brief description

Detailed explanation if needed.

Fixes #123
```

Types: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`

#### Pull Request Process

1. **Create feature branch** from `main`
2. **Write tests** for new functionality
3. **Ensure all tests pass**: `ctest --output-on-failure`
4. **Update documentation** if needed
5. **Submit PR** with description of changes

### Testing

All contributions must include tests:

- **Unit tests**: For individual functions
- **Integration tests**: For kernel pipelines
- **Performance tests**: For benchmarking

```cpp
// Example test structure
TEST(SYCLTransformTest, DCT8x8Accuracy) {
    // Setup
    auto& ctx = SYCLContext::instance();
    ASSERT_TRUE(ctx.initialize());

    // Test
    std::vector<int16_t> input(64), output(64);
    fdct8x8(ctx.queue(), input, output);

    // Verify
    for (int i = 0; i < 64; ++i) {
        EXPECT_NEAR(output[i], expected[i], tolerance);
    }
}
```

## Code Review Criteria

PRs are reviewed for:

- [ ] Code correctness
- [ ] Test coverage
- [ ] Performance impact
- [ ] Documentation updates
- [ ] Style consistency

## Questions?

Open an issue for questions or discussions.

---

Thank you for contributing!
