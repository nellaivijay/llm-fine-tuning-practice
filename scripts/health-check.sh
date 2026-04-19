#!/bin/bash

# Health Check Script for LLM Fine-Tuning Practice
# This script checks the health of the environment and services

set -e

echo "======================================"
echo "LLM Fine-Tuning Practice Health Check"
echo "======================================"

# Check Python
echo "Checking Python..."
if command -v python3 &> /dev/null; then
    python_version=$(python3 --version 2>&1)
    echo "✓ $python_version"
else
    echo "✗ Python not found"
    exit 1
fi

# Check CUDA
echo ""
echo "Checking CUDA..."
if command -v nvcc &> /dev/null; then
    cuda_version=$(nvcc --version | grep "release" | awk '{print $5}' | sed 's/,//')
    echo "✓ CUDA $cuda_version"
else
    echo "✗ CUDA not found"
fi

# Check GPU
echo ""
echo "Checking GPU..."
if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    gpu_name=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))")
    gpu_memory=$(python3 -c "import torch; print(torch.cuda.get_device_properties(0).total_memory / 1024**3)")
    echo "✓ GPU: $gpu_name"
    echo "✓ Memory: ${gpu_memory} GB"
else
    echo "✗ GPU not available or PyTorch not installed"
fi

# Check critical packages
echo ""
echo "Checking critical packages..."
packages=("torch" "transformers" "peft" "datasets" "accelerate")
for package in "${packages[@]}"; do
    if python3 -c "import $package" 2>/dev/null; then
        version=$(python3 -c "import $package; print($package.__version__)" 2>/dev/null || echo "unknown")
        echo "✓ $package ($version)"
    else
        echo "✗ $package not installed"
    fi
done

# Check directories
echo ""
echo "Checking directories..."
directories=("data" "models" "outputs" "checkpoints" "logs" "notebooks")
for dir in "${directories[@]}"; do
    if [ -d "$dir" ]; then
        echo "✓ $dir exists"
    else
        echo "✗ $dir missing"
    fi
done

# Check services
echo ""
echo "Checking services..."
if [ -f "logs/jupyter.pid" ]; then
    JUPYTER_PID=$(cat logs/jupyter.pid)
    if ps -p $JUPYTER_PID > /dev/null; then
        echo "✓ Jupyter Notebook running (PID: $JUPYTER_PID)"
    else
        echo "✗ Jupyter Notebook not running"
    fi
else
    echo "○ Jupyter Notebook not started"
fi

# Check disk space
echo ""
echo "Checking disk space..."
disk_usage=$(df -h . | tail -1 | awk '{print $5}' | sed 's/%//')
if [ $disk_usage -lt 90 ]; then
    echo "✓ Disk usage: ${disk_usage}%"
else
    echo "⚠ Disk usage: ${disk_usage}% (high)"
fi

# Check memory
echo ""
echo "Checking memory..."
if command -v free &> /dev/null; then
    mem_available=$(free -h | awk '/^Mem:/ {print $7}')
    echo "✓ Available memory: $mem_available"
fi

echo ""
echo "======================================"
echo "Health check completed"
echo "======================================"