#!/bin/bash
set -e  # Exit on error

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed. Please install uv first."
    exit 1
fi

echo "uv is installed, proceeding..."

# Check if .venv exists and remove it
if [ -d ".venv" ]; then
    echo "Removing existing .venv directory..."
    rm -rf .venv
fi

# Create new venv with Python 3.11
echo "Creating new virtual environment with Python 3.11..."
uv venv --python 3.11

echo "Virtual environment created successfully!"

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

# nvdiffrast-hip
if [ -d "./nvdiffrast-hip" ]; then
    rm -rf ./nvdiffrast-hip
fi

git clone https://github.com/CalebisGross/TRELLIS-AMD ./trellis
cd ./trellis
git checkout 2ccf54e8ff7aee0c519d37717bee6d95cf75357e
mv ./extensions/nvdiffrast-hip/ ../
cd ..
rm -rf ./trellis
cd ./nvdiffrast-hip
uv pip install . --no-build-isolation

cd ..

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

# Prepare nvdiffrec with hipify conversion
if [ -d "./nvdiffrec" ]; then
    rm -rf ./nvdiffrec
fi

echo "Cloning nvdiffrec repository..."
git clone -b renderutils https://github.com/JeffreyXiang/nvdiffrec.git
cd ./nvdiffrec

echo "Converting CUDA code to ROCm using hipify..."
# Find and convert all CUDA files to HIP
find . -type f \( -name "*.cu" -o -name "*.cuh" -o -name "*.cpp" -o -name "*.h" \) -exec hipify-perl {} -inplace \;

echo "Applying ROCm-specific fixes..."
# Fix __shfl_*_sync mask - ROCm requires 64-bit mask instead of 32-bit
find . -type f \( -name "*.cu" -o -name "*.hip" -o -name "*.cpp" -o -name "*.h" \) -exec sed -i 's/__shfl\([^(]*\)_sync(0xFFFFFFFF,/__shfl\1_sync(0xFFFFFFFFFFFFFFFFULL,/g' {} \;

# Fix incorrect hipify conversion: ATen/cuda/HIPContext.h should be ATen/hip/HIPContext.h
find . -type f \( -name "*.cpp" -o -name "*.h" \) -exec sed -i 's|ATen/cuda/HIPContext\.h|ATen/hip/HIPContext.h|g' {} \;

# Remove CUDAUtils.h include as it's not needed and not available in ROCm PyTorch
find . -type f \( -name "*.cpp" -o -name "*.h" \) -exec sed -i 's|#include <ATen/cuda/CUDAUtils\.h>|// #include <ATen/cuda/CUDAUtils.h> // Removed for ROCm|g' {} \;

# Fix getCurrentCUDAStream to getCurrentHIPStream for ROCm
find . -type f \( -name "*.cpp" -o -name "*.h" \) -exec sed -i 's/at::cuda::getCurrentCUDAStream/at::hip::getCurrentHIPStream/g' {} \;

# Update setup.py to use CppExtension instead of CUDAExtension
if [ -f "setup.py" ]; then
    # Change imports
    sed -i 's/from torch.utils.cpp_extension import BuildExtension, CUDAExtension/from torch.utils.cpp_extension import BuildExtension, CppExtension/g' setup.py

    # Change CUDAExtension to CppExtension
    sed -i 's/CUDAExtension(/CppExtension(/g' setup.py

    # Add ROCm include path to c_flags (needed for .cpp files that include HIP headers)
    sed -i "s/c_flags = \['-DNVDR_TORCH'\]/c_flags = ['-DNVDR_TORCH', '-I\/opt\/rocm\/include']/g" setup.py

    # Change nvcc_flags to hipcc_flags (variable name and references)
    sed -i "s/nvcc_flags = /hipcc_flags = /g" setup.py
    sed -i "s/nvcc_flags/hipcc_flags/g" setup.py

    # Change 'nvcc': to 'hipcc': in extra_compile_args
    sed -i "s/'nvcc':/'hipcc':/g" setup.py

    # Fix linker flags for ROCm
    sed -i "s/'-lcuda', '-lnvrtc'/'-lamdhip64'/g" setup.py
    sed -i "s/'cuda.lib', 'advapi32.lib', 'nvrtc.lib'/'amdhip64.lib', 'advapi32.lib'/g" setup.py

    # Note: We keep .cu extensions in setup.py even after hipify conversion
    # The files remain as .cu but contain HIP code, which hipcc can compile
    # hipify modifies files in-place, so torch_bindings.cpp stays as torch_bindings.cpp
fi

cd ..
echo "Installing nvdiffrec..."
uv pip install ./nvdiffrec --no-build-isolation