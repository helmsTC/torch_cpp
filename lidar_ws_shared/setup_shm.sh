#!/bin/bash

# Setup Script for MaskPLS Lidar Processor with Shared Memory
# This script sets up the environment and dependencies

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}MaskPLS ROS2 Shared Memory Setup${NC}"
echo -e "${GREEN}============================================${NC}"

# Function to print colored messages
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_step() {
    echo -e "\n${BLUE}>>> $1${NC}"
}

# Check if running as root
if [ "$EUID" -eq 0 ]; then 
   print_error "Please do not run this script as root/sudo"
   exit 1
fi

# Check Python version
print_step "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | grep -Po '(?<=Python )[0-9.]+')
REQUIRED_VERSION="3.8"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    print_error "Python $REQUIRED_VERSION or higher is required (found $PYTHON_VERSION)"
    exit 1
fi
print_info "Python $PYTHON_VERSION found ✓"

# Check ROS2 installation
print_step "Checking ROS2 installation..."
if [ -z "$ROS_DISTRO" ]; then
    print_error "ROS2 not sourced. Please source your ROS2 installation:"
    echo "  source /opt/ros/<distro>/setup.bash"
    exit 1
fi
print_info "ROS2 $ROS_DISTRO found ✓"

# Install system dependencies
print_step "Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    libpcl-dev \
    python3-pip \
    python3-venv \
    python3-dev \
    libeigen3-dev \
    libboost-all-dev

# Create Python virtual environment
print_step "Creating Python virtual environment..."
VENV_PATH="maskpls_venv"

if [ ! -d "$VENV_PATH" ]; then
    python3 -m venv $VENV_PATH
    print_info "Virtual environment created at $VENV_PATH"
else
    print_warning "Virtual environment already exists at $VENV_PATH"
fi

# Activate virtual environment
source $VENV_PATH/bin/activate

# Upgrade pip
print_step "Upgrading pip..."
pip install --upgrade pip wheel setuptools

# Install Python dependencies
print_step "Installing Python dependencies..."
pip install \
    numpy \
    torch==2.1.0 \
    pyyaml \
    easydict \
    tqdm \
    scipy \
    scikit-learn

# Check CUDA and install appropriate PyTorch
print_step "Checking CUDA..."
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d'.' -f1,2)
    print_info "CUDA $CUDA_VERSION detected"
    
    # Install PyTorch with CUDA support
    case "${CUDA_VERSION}" in
        "12.1"|"12.2"|"12.3"|"12.4")
            print_info "Installing PyTorch for CUDA 12.x"
            pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
            ;;
        "11.8")
            print_info "Installing PyTorch for CUDA 11.8"
            pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118
            ;;
        *)
            print_warning "CUDA version $CUDA_VERSION not directly supported"
            print_info "Installing CPU-only PyTorch"
            pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu
            ;;
    esac
else
    print_warning "CUDA not detected. Installing CPU-only PyTorch"
    pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu
fi

# Install MinkowskiEngine
print_step "Installing MinkowskiEngine..."
if python -c "import MinkowskiEngine" 2>/dev/null; then
    print_info "MinkowskiEngine already installed ✓"
else
    print_info "Building MinkowskiEngine from source..."
    
    # Install build dependencies
    pip install ninja
    
    # Clone and install MinkowskiEngine
    if [ ! -d "MinkowskiEngine" ]; then
        git clone https://github.com/NVIDIA/MinkowskiEngine.git
    fi
    
    cd MinkowskiEngine
    python setup.py install --blas=openblas --force_cuda
    cd ..
    
    # Verify installation
    if python -c "import MinkowskiEngine" 2>/dev/null; then
        print_info "MinkowskiEngine installed successfully ✓"
    else
        print_error "MinkowskiEngine installation failed"
        print_info "You may need to install it manually"
    fi
fi

# Create workspace structure
print_step "Setting up workspace structure..."
WORKSPACE="lidar_ws"

mkdir -p $WORKSPACE/src/lidar_processor/{src,include/lidar,launch,config,scripts,ckpt_to_pt}

# Copy files to workspace
print_info "Copying files to workspace..."

# List of files to copy
cat << EOF > files_to_copy.txt
src/lidar_shm.cpp -> $WORKSPACE/src/lidar_processor/src/
include/lidar/lidar_shm.h -> $WORKSPACE/src/lidar_processor/include/lidar/
CMakeLists.txt -> $WORKSPACE/src/lidar_processor/
package.xml -> $WORKSPACE/src/lidar_processor/
launch/lidar_shm.launch.py -> $WORKSPACE/src/lidar_processor/launch/
config/config_shm.yaml -> $WORKSPACE/src/lidar_processor/config/
scripts/maskpls_inference_server.py -> $WORKSPACE/src/lidar_processor/scripts/
ckpt_to_pt/original_pt_converter.py -> $WORKSPACE/src/lidar_processor/ckpt_to_pt/
EOF

print_warning "Please manually copy the files listed above to the workspace"

# Create a build script
print_step "Creating build script..."
cat > $WORKSPACE/build.sh << 'EOF'
#!/bin/bash
# Build script for lidar_processor

# Source ROS2
source /opt/ros/$ROS_DISTRO/setup.bash

# Build with Release mode for performance
colcon build \
    --packages-select lidar_processor \
    --cmake-args \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

# Source the workspace
source install/setup.bash

echo "Build complete! You can now run:"
echo "  ros2 launch lidar_processor lidar_shm.launch.py model_path:=/path/to/model.pt"
EOF

chmod +x $WORKSPACE/build.sh

# Create a run script
print_step "Creating run script..."
cat > $WORKSPACE/run.sh << 'EOF'
#!/bin/bash
# Run script for lidar_processor

# Check if model path is provided
if [ $# -eq 0 ]; then
    echo "Usage: ./run.sh /path/to/model.pt [additional_args]"
    exit 1
fi

MODEL_PATH=$1
shift  # Remove first argument

# Source ROS2 and workspace
source /opt/ros/$ROS_DISTRO/setup.bash
source install/setup.bash

# Activate Python virtual environment
source ../maskpls_venv/bin/activate

# Run the launch file
ros2 launch lidar_processor lidar_shm.launch.py \
    model_path:=$MODEL_PATH \
    use_cuda:=true \
    verbose:=true \
    auto_start_server:=true \
    $@
EOF

chmod +x $WORKSPACE/run.sh

# Create a test script
print_step "Creating test script..."
cat > test_shared_memory.py << 'EOF'
#!/usr/bin/env python3
"""Test script to verify shared memory setup"""

import numpy as np
from multiprocessing import shared_memory
import time

def test_shared_memory():
    print("Testing shared memory creation...")
    
    try:
        # Create test shared memory
        shm = shared_memory.SharedMemory(
            create=True,
            size=1024,
            name="/test_shm"
        )
        
        # Write test data
        data = np.ones(256, dtype=np.float32)
        shm_array = np.ndarray(data.shape, dtype=data.dtype, buffer=shm.buf)
        shm_array[:] = data[:]
        
        print("✓ Shared memory created successfully")
        
        # Clean up
        shm.close()
        shm.unlink()
        
        print("✓ Shared memory test passed")
        return True
        
    except Exception as e:
        print(f"✗ Shared memory test failed: {e}")
        return False

if __name__ == "__main__":
    test_shared_memory()
EOF

chmod +x test_shared_memory.py

# Run test
print_step "Testing shared memory..."
python test_shared_memory.py

# Print summary
echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}Setup Complete!${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
print_info "Workspace created at: $WORKSPACE"
print_info "Virtual environment at: $VENV_PATH"
echo ""
echo "Next steps:"
echo "  1. Copy your source files to the workspace (see files_to_copy.txt)"
echo "  2. Convert your model:"
echo "     cd $WORKSPACE/src/lidar_processor"
echo "     python ckpt_to_pt/original_pt_converter.py --checkpoint /path/to/checkpoint.ckpt --output model.pt"
echo ""
echo "  3. Build the workspace:"
echo "     cd $WORKSPACE"
echo "     ./build.sh"
echo ""
echo "  4. Run the node:"
echo "     ./run.sh /path/to/model.pt"
echo ""
print_warning "Remember to activate the virtual environment when running Python scripts:"
echo "  source $VENV_PATH/bin/activate"
echo ""

# Create requirements.txt for reference
cat > requirements.txt << EOF
# MaskPLS Dependencies
numpy>=1.19.0
torch>=2.0.0
pyyaml
easydict
tqdm
scipy
scikit-learn
# MinkowskiEngine (install separately)
EOF

print_info "Python requirements saved to requirements.txt"
