#!/bin/bash
set -e  # Exit on error

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

sudo apt install -y libjpeg-dev

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

# Activate venv and install PyTorch with ROCm 6.4
echo "Installing PyTorch with ROCm 6.4 support..."
source .venv/bin/activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.4

echo "Installing additional dependencies..."
uv pip install imageio imageio-ffmpeg tqdm easydict opencv-python-headless ninja trimesh transformers gradio==6.0.1 tensorboard pandas lpips zstandard
uv pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8
uv pip install pillow-simd
uv pip install kornia timm
uv pip install flash-attn==2.8.3

echo "Dependencies installed successfully!"

# Prepare nvdiffrast
if [ -d "./trellis" ]; then
    rm -rf ./trellis
fi

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

# CuMesh with hipify conversion
if [ -d "./CuMesh" ]; then
    rm -rf ./CuMesh
fi

echo "Cloning CuMesh repository..."
git clone https://github.com/JeffreyXiang/CuMesh.git --recursive
cd ./CuMesh

echo "Converting CUDA code to ROCm using hipify..."
# Hipify main sources
find src -name "*.cu" -o -name "*.cpp" | xargs -I {} hipify-perl {} -inplace
# Hipify cubvh sources
find third_party/cubvh/src -name "*.cu" -o -name "*.cpp" | xargs -I {} hipify-perl {} -inplace
# Hipify headers (exclude Eigen - it already supports HIP)
find . -name "*.cuh" -o -name "*.h" | grep -v ".prehip" | grep -v "third_party/cubvh/third_party/eigen" | xargs -I {} hipify-perl {} -inplace

echo "Applying ROCm-specific fixes..."
# Fix __shfl_*_sync mask - ROCm requires 64-bit mask
find . -type f \( -name "*.cu" -o -name "*.hip" -o -name "*.cpp" -o -name "*.h" -o -name "*.cuh" \) -exec sed -i 's/__shfl\([^(]*\)_sync(0xFFFFFFFF,/__shfl\1_sync(0xFFFFFFFFFFFFFFFFULL,/g' {} \;

# Fix HIPContext includes
find . -type f \( -name "*.cpp" -o -name "*.h" \) -exec sed -i 's|ATen/cuda/HIPContext\.h|ATen/hip/HIPContext.h|g' {} \;

# Remove CUDAUtils includes
find . -type f \( -name "*.cpp" -o -name "*.h" \) -exec sed -i 's|#include <ATen/cuda/CUDAUtils\.h>|// #include <ATen/cuda/CUDAUtils.h> // Removed for ROCm|g' {} \;

# Fix stream API
find . -type f \( -name "*.cpp" -o -name "*.h" \) -exec sed -i 's/at::cuda::getCurrentCUDAStream/at::hip::getCurrentHIPStream/g' {} \;

# Fix cuda::std namespace to std (thrust doesn't have tuple in ROCm)
sed -i 's/::cuda::std::/::std::/g' src/clean_up.cu

# Apply comprehensive fix to clean_up.cu for ROCm compatibility
echo "Applying ROCm compatibility fixes to clean_up.cu..."

# Add thrust includes after hipcub include
sed -i '/#include <hipcub\/hipcub.hpp>/a #include <thrust/sort.h>\n#include <thrust/device_ptr.h>' src/clean_up.cu

# Replace int3_decomposer with int3_comparator
sed -i '/struct int3_decomposer/,/^};/c\
// Comparator for int3 to sort lexicographically (x, y, z)\
struct int3_comparator\
{\
    __host__ __device__ bool operator()(const int3\& a, const int3\& b) const\
    {\
        if (a.x != b.x) return a.x < b.x;\
        if (a.y != b.y) return a.y < b.y;\
        return a.z < b.z;\
    }\
};' src/clean_up.cu

# Use Python to replace DeviceRadixSort with thrust (handles multiline better)
python3 << 'PYTHON_SCRIPT'
import re

with open('src/clean_up.cu', 'r') as f:
    content = f.read()

# Pattern to match the two DeviceRadixSort calls plus the resize between them
pattern = r'    CUDA_CHECK\(hipcub::DeviceRadixSort::SortPairs\(\s*nullptr,.*?\)\);\s*this->cub_temp_storage\.resize\(temp_storage_bytes\);\s*CUDA_CHECK\(hipcub::DeviceRadixSort::SortPairs\(.*?int3_decomposer\{\}\s*\)\);'

replacement = '''    // Use thrust::sort_by_key instead of hipcub with decomposer (not supported in ROCm)
    CUDA_CHECK(hipMemcpy(cu_sorted_faces_output, cu_sorted_faces, F * sizeof(int3), hipMemcpyDeviceToDevice));
    CUDA_CHECK(hipMemcpy(cu_sorted_indices_output, cu_sorted_face_indices, F * sizeof(int), hipMemcpyDeviceToDevice));

    thrust::device_ptr<int3> faces_ptr(cu_sorted_faces_output);
    thrust::device_ptr<int> indices_ptr(cu_sorted_indices_output);
    thrust::sort_by_key(faces_ptr, faces_ptr + F, indices_ptr, int3_comparator());

    CUDA_CHECK(hipDeviceSynchronize());'''

content = re.sub(pattern, replacement, content, flags=re.DOTALL)

with open('src/clean_up.cu', 'w') as f:
    f.write(content)
PYTHON_SCRIPT

# Add __host__ to Vec3f constructors for host-side compatibility (both declarations and implementations)
# Fix declarations in class
sed -i 's/__device__ __forceinline__ Vec3f();/__host__ __device__ __forceinline__ Vec3f();/g' src/dtypes.cuh
sed -i 's/__device__ __forceinline__ Vec3f(float x, float y, float z);/__host__ __device__ __forceinline__ Vec3f(float x, float y, float z);/g' src/dtypes.cuh
sed -i 's/__device__ __forceinline__ Vec3f(float3 v);/__host__ __device__ __forceinline__ Vec3f(float3 v);/g' src/dtypes.cuh
# Fix implementations
sed -i 's/__device__ __forceinline__ Vec3f::Vec3f/__host__ __device__ __forceinline__ Vec3f::Vec3f/g' src/dtypes.cuh

# Update CuMesh setup.py
if [ -f "setup.py" ]; then
    # Change CUDAExtension to CppExtension
    sed -i 's/from torch.utils.cpp_extension import CUDAExtension, BuildExtension, IS_HIP_EXTENSION/from torch.utils.cpp_extension import CppExtension, BuildExtension, IS_HIP_EXTENSION/g' setup.py
    sed -i 's/CUDAExtension(/CppExtension(/g' setup.py

    # Change nvcc to hipcc in extra_compile_args
    sed -i 's/"nvcc":/"hipcc":/g' setup.py

    # Add ROCm include path for C++ files
    sed -i 's/"cxx": \["-O3", "-std=c++17"\]/"cxx": ["-O3", "-std=c++17", "-I\/opt\/rocm\/include"]/g' setup.py
fi

# Update cubvh setup.py
if [ -f "third_party/cubvh/setup.py" ]; then
    sed -i 's/from torch.utils.cpp_extension import BuildExtension, CUDAExtension/from torch.utils.cpp_extension import BuildExtension, CppExtension/g' third_party/cubvh/setup.py
    sed -i 's/CUDAExtension(/CppExtension(/g' third_party/cubvh/setup.py
    sed -i "s/'nvcc':/'hipcc':/g" third_party/cubvh/setup.py
fi

# Fix Eigen for HIP in cubvh
echo "Fixing Eigen for ROCm/HIP compatibility..."
# Fix Eigen Core to detect HIP using __HIP_PLATFORM_AMD__
if [ -f "third_party/cubvh/third_party/eigen/Eigen/Core" ]; then
    sed -i 's|#if defined(EIGEN_CUDACC)|#if defined(__HIPCC__) \|\| defined(__HIP_PLATFORM_AMD__)\
#include <hip/hip_runtime.h>\
#elif defined(EIGEN_CUDACC)|g' third_party/cubvh/third_party/eigen/Eigen/Core
fi

# Fix Eigen Meta.h to use correct HIP math constants path
if [ -f "third_party/cubvh/third_party/eigen/Eigen/src/Core/util/Meta.h" ]; then
    sed -i 's|#include "Eigen/src/Core/arch/HIP/hcc/hip/hip_math_constants.h"|#include <hip/hip_math_constants.h>|g' third_party/cubvh/third_party/eigen/Eigen/src/Core/util/Meta.h
fi

# Fix cubvh thrust and stream APIs
echo "Fixing cubvh thrust and stream APIs..."
find third_party/cubvh -type f \( -name "*.cu" -o -name "*.cpp" -o -name "*.h" -o -name "*.cuh" \) -exec sed -i 's/thrust::cuda::/thrust::hip::/g' {} \;
find third_party/cubvh -type f \( -name "*.cu" -o -name "*.cpp" -o -name "*.h" -o -name "*.cuh" \) -exec sed -i 's/at::cuda::getCurrentCUDAStream/at::hip::getCurrentHIPStream/g' {} \;

cd ..
echo "Installing CuMesh..."
uv pip install ./CuMesh --no-build-isolation

# FlexGEMM with hipify conversion
if [ -d "./FlexGEMM" ]; then
    rm -rf ./FlexGEMM
fi

echo "Cloning FlexGEMM repository..."
git clone https://github.com/JeffreyXiang/FlexGEMM.git
cd ./FlexGEMM

echo "Converting CUDA code to ROCm using hipify..."
find flex_gemm/kernels/cuda -type f \( -name "*.cu" -o -name "*.cpp" -o -name "*.cuh" -o -name "*.h" \) | xargs -I {} hipify-perl {} -inplace

echo "Applying ROCm-specific fixes to FlexGEMM..."
# Update setup.py to use CppExtension instead of CUDAExtension
if [ -f "setup.py" ]; then
    sed -i 's/from torch.utils.cpp_extension import CUDAExtension, BuildExtension/from torch.utils.cpp_extension import CppExtension, BuildExtension/g' setup.py
    sed -i 's/CUDAExtension(/CppExtension(/g' setup.py
    sed -i 's/"nvcc":/"hipcc":/g' setup.py
fi

cd ..
echo "Installing FlexGEMM..."
uv pip install ./FlexGEMM --no-build-isolation

# o-voxel with hipify conversion
if [ -d "./o-voxel-hip" ]; then
    rm -rf ./o-voxel-hip
fi

echo "Copying o-voxel to o-voxel-hip..."
cp -r ./o-voxel ./o-voxel-hip
cd ./o-voxel-hip

echo "Converting CUDA code to ROCm using hipify..."
# Hipify main sources
find src -type f \( -name "*.cu" -o -name "*.cpp" -o -name "*.cuh" -o -name "*.h" \) | xargs -I {} hipify-perl {} -inplace

echo "Applying ROCm-specific fixes to o-voxel..."
# Fix __shfl_*_sync mask - ROCm requires 64-bit mask
find . -type f \( -name "*.cu" -o -name "*.hip" -o -name "*.cpp" -o -name "*.h" -o -name "*.cuh" \) -exec sed -i 's/__shfl\([^(]*\)_sync(0xFFFFFFFF,/__shfl\1_sync(0xFFFFFFFFFFFFFFFFULL,/g' {} \;

# Fix HIPContext includes
find . -type f \( -name "*.cpp" -o -name "*.h" \) -exec sed -i 's|ATen/cuda/HIPContext\.h|ATen/hip/HIPContext.h|g' {} \;

# Remove CUDAUtils includes
find . -type f \( -name "*.cpp" -o -name "*.h" \) -exec sed -i 's|#include <ATen/cuda/CUDAUtils\.h>|// #include <ATen/cuda/CUDAUtils.h> // Removed for ROCm|g' {} \;

# Fix stream API
find . -type f \( -name "*.cpp" -o -name "*.h" \) -exec sed -i 's/at::cuda::getCurrentCUDAStream/at::hip::getCurrentHIPStream/g' {} \;

# Remove local struct definitions that conflict with HIP types
# HIP already provides float3, int3, int4 but NOT bool3
sed -i '/^struct float3 {float x, y, z;/d' src/convert/flexible_dual_grid.cpp
sed -i '/^struct int3 {int x, y, z;/d' src/convert/flexible_dual_grid.cpp
sed -i '/^struct int4 {int x, y, z, w;/d' src/convert/flexible_dual_grid.cpp
# Keep bool3 as it's not provided by HIP

# Update setup.py to use CppExtension instead of CUDAExtension
if [ -f "setup.py" ]; then
    # Change CUDAExtension to CppExtension
    sed -i 's/from torch.utils.cpp_extension import CUDAExtension, BuildExtension, IS_HIP_EXTENSION/from torch.utils.cpp_extension import CppExtension, BuildExtension, IS_HIP_EXTENSION/g' setup.py
    sed -i 's/CUDAExtension(/CppExtension(/g' setup.py

    # Change nvcc to hipcc in extra_compile_args
    sed -i 's/"nvcc":/"hipcc":/g' setup.py

    # Add ROCm include path and Eigen path for C++ files
    # Use Python to dynamically insert the path
    python3 << 'PYTHON_SCRIPT'
import re
with open('setup.py', 'r') as f:
    content = f.read()

# Replace cxx flags
content = re.sub(
    r'"cxx": \["-O3", "-std=c\+\+17"\]',
    '"cxx": ["-I/opt/rocm/include", f"-I{os.path.join(ROOT, \'third_party/eigen\')}", "-O3", "-std=c++17"]',
    content
)

with open('setup.py', 'w') as f:
    f.write(content)
PYTHON_SCRIPT
fi

# Download Eigen if not present
if [ ! -d "third_party/eigen" ] || [ -z "$(ls -A third_party/eigen 2>/dev/null)" ]; then
    echo "Downloading Eigen library..."
    mkdir -p third_party
    rm -rf third_party/eigen
    git clone --depth 1 https://gitlab.com/libeigen/eigen.git third_party/eigen
fi

# Update pyproject.toml to use local CuMesh and FlexGEMM
if [ -f "pyproject.toml" ]; then
    # Comment out git dependencies - we'll use already installed local versions
    sed -i 's|"cumesh @ git+https://github.com/JeffreyXiang/CuMesh.git",|# "cumesh @ git+https://github.com/JeffreyXiang/CuMesh.git",  # Use locally installed version|g' pyproject.toml
    sed -i 's|"flex_gemm @ git+https://github.com/JeffreyXiang/FlexGEMM.git",|# "flex_gemm @ git+https://github.com/JeffreyXiang/FlexGEMM.git",  # Use locally installed version|g' pyproject.toml
fi

cd ..
echo "Installing o-voxel-hip..."
uv pip install ./o-voxel-hip --no-build-isolation