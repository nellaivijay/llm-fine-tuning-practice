#!/bin/bash

# LLM Fine-Tuning Practice Setup Script
# This script sets up the environment for fine-tuning

set -e

echo "======================================"
echo "LLM Fine-Tuning Practice Setup"
echo "======================================"

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Check CUDA
echo "Checking CUDA installation..."
if command -v nvcc &> /dev/null; then
    nvcc --version
else
    echo "Warning: CUDA not found. GPU training may not work."
fi

# Check GPU
echo "Checking GPU availability..."
if python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())" 2>/dev/null; then
    python3 -c "import torch; print('GPU name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
else
    echo "Warning: PyTorch not installed or CUDA not available"
fi

# Create virtual environment
echo "Creating virtual environment..."
if [ ! -d "llm-env" ]; then
    python3 -m venv llm-env
    echo "Virtual environment created"
else
    echo "Virtual environment already exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source llm-env/bin/activate

# Install PyTorch
echo "Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
echo "Installing project dependencies..."
pip install -r requirements.txt

# Create directories
echo "Creating necessary directories..."
mkdir -p data models outputs checkpoints logs config

# Copy environment template
echo "Setting up environment configuration..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "Created .env file from template"
    echo "Please edit .env with your configuration"
else
    echo ".env file already exists"
fi

# Set .gitignore for notebooks
echo "Updating .gitignore for notebooks..."
if ! grep -q "!notebooks/*.ipynb" .gitignore; then
    sed -i 's/\.ipynb_checkpoints\/$/\.ipynb_checkpoints\/\n!notebooks\/\*\.ipynb/' .gitignore
fi

# Test imports
echo "Testing critical imports..."
python3 -c "
import torch
import transformers
import peft
print('✓ PyTorch imported successfully')
print('✓ Transformers imported successfully')
print('✓ PEFT imported successfully')
"

echo ""
echo "======================================"
echo "Setup completed successfully!"
echo "======================================"
echo ""
echo "Next steps:"
echo "1. Edit .env with your configuration"
echo "2. Activate environment: source llm-env/bin/activate"
echo "3. Start with Chapter 1 notebooks"
echo ""
echo "To verify GPU access, run:"
echo "python3 -c 'import torch; print(torch.cuda.is_available())'"
echo ""