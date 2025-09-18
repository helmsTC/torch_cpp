#!/bin/bash

# Script to fix NVIDIA driver/library version mismatch
# Run with sudo: sudo bash fix_nvidia_mismatch.sh

set -e

echo "================================================"
echo "NVIDIA Driver Version Mismatch Fix Script"
echo "================================================"
echo ""

# Function to print colored output
print_status() {
    echo -e "\033[1;34m[INFO]\033[0m $1"
}

print_success() {
    echo -e "\033[1;32m[SUCCESS]\033[0m $1"
}

print_error() {
    echo -e "\033[1;31m[ERROR]\033[0m $1"
}

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    print_error "Please run this script with sudo"
    exit 1
fi

# 1. Check current driver status
print_status "Checking current NVIDIA driver status..."
echo "Current kernel module version:"
cat /proc/driver/nvidia/version 2>/dev/null || echo "Kernel module not loaded"
echo ""
echo "NVIDIA SMI version:"
nvidia-smi --version 2>/dev/null | grep "NVIDIA-SMI" || echo "nvidia-smi not accessible"
echo ""

# 2. Stop all processes using NVIDIA GPUs
print_status "Stopping processes using NVIDIA GPUs..."

# Stop Docker containers using GPUs
if command -v docker &> /dev/null; then
    print_status "Stopping Docker containers with GPUs..."
    docker ps -q | xargs -r docker stop 2>/dev/null || true
fi

# Stop NVIDIA persistence daemon
print_status "Stopping NVIDIA persistence daemon..."
systemctl stop nvidia-persistenced 2>/dev/null || true
pkill -f nvidia-persistenced 2>/dev/null || true

# Kill any remaining NVIDIA processes
print_status "Terminating NVIDIA processes..."
lsof /dev/nvidia* 2>/dev/null | awk '{print $2}' | tail -n +2 | sort -u | xargs -r kill -9 2>/dev/null || true
pkill -f nvidia 2>/dev/null || true

# 3. Unload NVIDIA kernel modules
print_status "Unloading NVIDIA kernel modules..."
rmmod nvidia_drm 2>/dev/null || true
rmmod nvidia_modeset 2>/dev/null || true
rmmod nvidia_uvm 2>/dev/null || true
rmmod nvidia 2>/dev/null || true

# 4. Reload NVIDIA kernel modules
print_status "Reloading NVIDIA kernel modules..."
modprobe nvidia
modprobe nvidia_uvm
modprobe nvidia_modeset
modprobe nvidia_drm

# 5. Restart NVIDIA services
print_status "Restarting NVIDIA services..."

# Start NVIDIA persistence daemon
systemctl start nvidia-persistenced 2>/dev/null || nvidia-persistenced --persistence-mode || true

# Restart Docker daemon if it exists
if command -v docker &> /dev/null; then
    print_status "Restarting Docker daemon..."
    systemctl restart docker 2>/dev/null || service docker restart 2>/dev/null || true
    sleep 2
fi

# 6. Verify the fix
print_status "Verifying NVIDIA driver status..."
echo ""

if nvidia-smi &>/dev/null; then
    print_success "NVIDIA SMI is working!"
    echo ""
    nvidia-smi
    echo ""
    
    # Test Docker GPU access
    if command -v docker &> /dev/null; then
        print_status "Testing Docker GPU access..."
        if docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi &>/dev/null; then
            print_success "Docker can access GPUs successfully!"
        else
            print_error "Docker GPU access test failed. You may need to:"
            echo "  1. Ensure nvidia-docker2 is installed"
            echo "  2. Check Docker daemon configuration"
            echo "  3. Restart your system"
        fi
    fi
else
    print_error "NVIDIA SMI is still not working. You may need to:"
    echo "  1. Reinstall NVIDIA drivers"
    echo "  2. Reboot your system"
    echo "  3. Check for driver conflicts"
fi

echo ""
echo "================================================"
echo "Script completed!"
echo "================================================"
echo ""
echo "If the issue persists, try:"
echo "  1. Rebooting your system (recommended)"
echo "  2. Reinstalling NVIDIA drivers:"
echo "     sudo apt-get purge nvidia-* cuda-*"
echo "     sudo apt-get autoremove"
echo "     sudo ubuntu-drivers autoinstall"
echo "  3. Checking for multiple NVIDIA driver installations:"
echo "     dpkg -l | grep nvidia"
echo ""
