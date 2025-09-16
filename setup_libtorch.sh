#!/bin/bash

# LibTorch Setup Script for ROS2 Lidar Processor
# This script downloads and sets up LibTorch for C++ development

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}===================================${NC}"
echo -e "${GREEN}LibTorch Setup Script${NC}"
echo -e "${GREEN}===================================${NC}"

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

# Detect OS and architecture
OS=$(uname -s)
ARCH=$(uname -m)

print_info "Detected OS: $OS"
print_info "Detected Architecture: $ARCH"

# Check for CUDA
CUDA_AVAILABLE=false
CUDA_VERSION=""

if command -v nvidia-smi &> /dev/null; then
    CUDA_AVAILABLE=true
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d'.' -f1,2)
    print_info "CUDA detected: Version $CUDA_VERSION"
else
    print_warning "CUDA not detected. Will download CPU-only version."
fi

# Set LibTorch download URL based on system configuration
LIBTORCH_URL=""

if [[ "$OS" == "Linux" ]]; then
    if [[ "$CUDA_AVAILABLE" == true ]]; then
        # CUDA versions - adjust based on your CUDA version
        case "${CUDA_VERSION}" in
            "12.1"|"12.2"|"12.3"|"12.4")
                LIBTORCH_URL="https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcu121.zip"
                print_info "Downloading LibTorch for CUDA 12.x"
                ;;
            "11.8")
                LIBTORCH_URL="https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcu118.zip"
                print_info "Downloading LibTorch for CUDA 11.8"
                ;;
            *)
                print_warning "CUDA version $CUDA_VERSION not directly supported. Downloading CPU version."
                LIBTORCH_URL="https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcpu.zip"
                ;;
        esac
    else
        # CPU only version
        LIBTORCH_URL="https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcpu.zip"
        print_info "Downloading LibTorch CPU version"
    fi
elif [[ "$OS" == "Darwin" ]]; then
    # macOS
    if [[ "$ARCH" == "arm64" ]]; then
        # Apple Silicon
        LIBTORCH_URL="https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-2.1.0.zip"
        print_info "Downloading LibTorch for macOS ARM64 (Apple Silicon)"
    else
        # Intel Mac
        LIBTORCH_URL="https://download.pytorch.org/libtorch/cpu/libtorch-macos-x86_64-2.1.0.zip"
        print_info "Downloading LibTorch for macOS x86_64"
    fi
else
    print_error "Unsupported operating system: $OS"
    exit 1
fi

# Create workspace directory structure
WORKSPACE_DIR="lidar_ws"
print_info "Creating workspace directory: $WORKSPACE_DIR"

mkdir -p $WORKSPACE_DIR/src/lidar_processor/{src,include/lidar,launch,config}
cd $WORKSPACE_DIR/src/lidar_processor

# Download LibTorch if not already present
if [ -d "libtorch" ]; then
    print_warning "LibTorch directory already exists. Skipping download."
else
    print_info "Downloading LibTorch..."
    wget -q --show-progress "$LIBTORCH_URL" -O libtorch.zip
    
    print_info "Extracting LibTorch..."
    unzip -q libtorch.zip
    rm libtorch.zip
    
    print_info "LibTorch extracted successfully!"
fi

# Copy the source files (assuming they're in the current directory)
cd ../../..

print_info "Setting up source files..."

# Create launch directory and a sample launch file
mkdir -p $WORKSPACE_DIR/src/lidar_processor/launch
cat > $WORKSPACE_DIR/src/lidar_processor/launch/lidar_processor.launch.py << 'EOF'
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'model_path',
            default_value='',
            description='Path to the PyTorch model file (.pt)'
        ),
        DeclareLaunchArgument(
            'input_topic',
            default_value='/points',
            description='Input PointCloud2 topic'
        ),
        DeclareLaunchArgument(
            'use_cuda',
            default_value='false',
            description='Use CUDA if available'
        ),
        DeclareLaunchArgument(
            'batch_size',
            default_value='1',
            description='Batch size for inference'
        ),
        
        Node(
            package='lidar_processor',
            executable='lidar_processor_node',
            name='lidar_processor',
            output='screen',
            parameters=[{
                'model_path': LaunchConfiguration('model_path'),
                'input_topic': LaunchConfiguration('input_topic'),
                'use_cuda': LaunchConfiguration('use_cuda'),
                'batch_size': LaunchConfiguration('batch_size'),
            }]
        ),
    ])
EOF

# Create a sample config file
mkdir -p $WORKSPACE_DIR/src/lidar_processor/config
cat > $WORKSPACE_DIR/src/lidar_processor/config/lidar_processor.yaml << 'EOF'
lidar_processor:
  ros__parameters:
    model_path: ""  # Path to your .pt model file
    input_topic: "/points"
    use_cuda: false
    batch_size: 1
EOF

# Create a README
cat > $WORKSPACE_DIR/src/lidar_processor/README.md << 'EOF'
# Lidar Processor ROS2 Package

## Description
This package provides a ROS2 node that subscribes to PointCloud2 messages and processes them using PyTorch models via LibTorch.

## Installation

1. Install ROS2 (Humble/Iron/Rolling)
2. Run the setup script: `./setup_libtorch.sh`
3. Copy your source files to the appropriate directories
4. Build the workspace:
   ```bash
   cd lidar_ws
   colcon build --packages-select lidar_processor
   source install/setup.bash
   ```

## Usage

### Run with launch file:
```bash
ros2 launch lidar_processor lidar_processor.launch.py model_path:=/path/to/model.pt
```

### Run node directly:
```bash
ros2 run lidar_processor lidar_processor_node --ros-args \
  -p model_path:=/path/to/your/model.pt \
  -p input_topic:=/velodyne_points \
  -p use_cuda:=true
```

### Parameters:
- `model_path`: Path to the PyTorch model file (.pt)
- `input_topic`: Input PointCloud2 topic (default: /points)
- `use_cuda`: Use CUDA if available (default: false)
- `batch_size`: Batch size for inference (default: 1)

## Creating a PyTorch Model

Example Python script to create a compatible .pt model:

```python
import torch
import torch.nn as nn

class PointNetExample(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 64)  # 4 features: x, y, z, intensity
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 4)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Create and save model
model = PointNetExample()
model.eval()

# Trace the model
example_input = torch.randn(1, 1000, 4)  # batch_size, num_points, features
traced_model = torch.jit.trace(model, example_input)
traced_model.save("model.pt")
```

## Topics

### Subscribed:
- `<input_topic>` (sensor_msgs/PointCloud2): Input point cloud

### Published:
- `/processed_points` (sensor_msgs/PointCloud2): Processed point cloud

## Dependencies
- ROS2 (Humble or newer)
- LibTorch (automatically downloaded by setup script)
- C++17 compiler
EOF

print_info "Creating build script..."
cat > $WORKSPACE_DIR/build.sh << 'EOF'
#!/bin/bash
colcon build --packages-select lidar_processor --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
EOF
chmod +x $WORKSPACE_DIR/build.sh

# Final instructions
echo ""
echo -e "${GREEN}===================================${NC}"
echo -e "${GREEN}Setup Complete!${NC}"
echo -e "${GREEN}===================================${NC}"
echo ""
print_info "Workspace created at: $WORKSPACE_DIR"
print_info "Next steps:"
echo "  1. Copy the provided source files to:"
echo "     - src/lidar.cpp → $WORKSPACE_DIR/src/lidar_processor/src/"
echo "     - include/lidar/lidar.h → $WORKSPACE_DIR/src/lidar_processor/include/lidar/"
echo "     - CMakeLists.txt → $WORKSPACE_DIR/src/lidar_processor/"
echo "     - package.xml → $WORKSPACE_DIR/src/lidar_processor/"
echo ""
echo "  2. Build the workspace:"
echo "     cd $WORKSPACE_DIR"
echo "     ./build.sh"
echo ""
echo "  3. Run the node:"
echo "     ros2 launch lidar_processor lidar_processor.launch.py model_path:=/path/to/model.pt"
echo ""
print_warning "Make sure you have ROS2 sourced before building!"
print_warning "Example: source /opt/ros/humble/setup.bash"