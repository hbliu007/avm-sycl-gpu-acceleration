#!/bin/bash
# AVM SYCL GPU Acceleration - One-Line Installer for Linux
# Usage: curl -fsSL https://raw.githubusercontent.com/hbliu007/avm-sycl-gpu-acceleration/main/install.sh | bash

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

REPO_URL="https://github.com/hbliu007/avm-sycl-gpu-acceleration.git"
INSTALL_DIR="${INSTALL_DIR:-$HOME/avm-sycl-gpu-acceleration}"

echo -e "${BLUE}"
echo "╔════════════════════════════════════════════════════════════╗"
echo "║     🎬 AVM SYCL GPU Acceleration Installer                ║"
echo "║     Cross-platform GPU acceleration for AV2 codec          ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Check for existing Intel oneAPI installation
check_oneapi() {
    if [ -f "/opt/intel/oneapi/setvars.sh" ]; then
        echo -e "${GREEN}✓ Intel oneAPI found at /opt/intel/oneapi${NC}"
        return 0
    else
        echo -e "${YELLOW}⚠ Intel oneAPI not found${NC}"
        return 1
    fi
}

# Install Intel oneAPI
install_oneapi() {
    echo -e "${BLUE}Installing Intel oneAPI DPC++...${NC}"

    # Add Intel repository
    wget -q -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | \
        gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null 2>&1

    echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | \
        sudo tee /etc/apt/sources.list.d/oneAPI.list > /dev/null

    # Install DPC++
    sudo apt-get update -qq
    sudo apt-get install -y -qq intel-oneapi-dpcpp-cpp intel-oneapi-mkl

    echo -e "${GREEN}✓ Intel oneAPI installed successfully${NC}"
}

# Clone repository
clone_repo() {
    echo -e "${BLUE}Cloning repository...${NC}"

    if [ -d "$INSTALL_DIR" ]; then
        echo -e "${YELLOW}Directory $INSTALL_DIR exists. Updating...${NC}"
        cd "$INSTALL_DIR"
        git pull
    else
        git clone "$REPO_URL" "$INSTALL_DIR"
        cd "$INSTALL_DIR"
    fi

    echo -e "${GREEN}✓ Repository ready at $INSTALL_DIR${NC}"
}

# Build project
build_project() {
    echo -e "${BLUE}Building project...${NC}"

    source /opt/intel/oneapi/setvars.sh

    mkdir -p build && cd build
    cmake .. -DCMAKE_CXX_COMPILER=icpx \
             -DCMAKE_BUILD_TYPE=Release \
             -DAVM_ENABLE_SYCL=ON \
             -DAVM_BUILD_TESTS=ON \
             -DAVM_BUILD_EXAMPLES=ON

    make -j$(nproc)

    echo -e "${GREEN}✓ Build completed${NC}"
}

# Run tests
run_tests() {
    echo -e "${BLUE}Running tests...${NC}"

    ctest --output-on-failure

    echo -e "${GREEN}✓ All tests passed${NC}"
}

# Main installation flow
main() {
    echo -e "${YELLOW}Step 1/5: Checking prerequisites...${NC}"
    if ! check_oneapi; then
        read -p "Install Intel oneAPI DPC++? [Y/n] " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]; then
            install_oneapi
        else
            echo -e "${RED}Cannot proceed without Intel oneAPI. Exiting.${NC}"
            exit 1
        fi
    fi

    echo -e "${YELLOW}Step 2/5: Cloning repository...${NC}"
    clone_repo

    echo -e "${YELLOW}Step 3/5: Building project...${NC}"
    build_project

    echo -e "${YELLOW}Step 4/5: Running tests...${NC}"
    run_tests

    echo -e "${YELLOW}Step 5/5: Installation complete!${NC}"

    echo ""
    echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║              ✅ Installation Successful!                   ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "📁 Install location: ${BLUE}$INSTALL_DIR${NC}"
    echo ""
    echo -e "Quick start commands:"
    echo -e "  ${BLUE}cd $INSTALL_DIR/build${NC}"
    echo -e "  ${BLUE}source /opt/intel/oneapi/setvars.sh${NC}"
    echo -e "  ${BLUE}./tests/sycl_context_test${NC}"
    echo ""
    echo -e "📚 Documentation: ${BLUE}https://github.com/hbliu007/avm-sycl-gpu-acceleration${NC}"
    echo ""
}

main "$@"
