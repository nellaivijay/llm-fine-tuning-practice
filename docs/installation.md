---
title: Installation
description: Complete guide to setting up LLM Fine-Tuning Practice environment
---

# Installation Guide

This guide will help you set up the LLM Fine-Tuning Practice environment on your local machine.

## Prerequisites

### Required Software

- **Python**: Version 3.8 or higher
- **CUDA**: Version 11.8 or higher (for GPU training)
- **Git**: For cloning the repository

### System Requirements

- **Operating System**: Linux, macOS, or Windows with WSL2
- **GPU**: NVIDIA GPU with 8GB+ VRAM (recommended)
- **RAM**: Minimum 16GB (32GB recommended)
- **Disk Space**: Minimum 50GB free space
- **CPU**: Multi-core processor recommended

### Optional Requirements

- **Docker**: For containerized environment
- **Google Colab**: For cloud-based training
- **Cloud Account**: AWS/GCP/Azure for cloud GPUs

## Installation Steps

### Step 1: Install NVIDIA Drivers

#### Linux (Ubuntu/Debian)
```bash
sudo apt-get update
sudo apt-get install nvidia-driver-535
nvidia-smi
```

#### Windows
Download and install from [NVIDIA website](https://www.nvidia.com/Download/index.aspx)

### Step 2: Install CUDA

#### Linux
```bash
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_linux.run
sudo sh cuda_11.8.0_linux.run
echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
nvcc --version
```

### Step 3: Clone the Repository

```bash
git clone https://github.com/nellaivijay/llm-fine-tuning-practice.git
cd llm-fine-tuning-practice
```

### Step 4: Create Virtual Environment

```bash
python3 -m venv llm-env
source llm-env/bin/activate
```

### Step 5: Install Dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### Step 6: Configure Environment

```bash
cp .env.example .env
# Edit .env with your settings
```

### Step 7: Verify Installation

```bash
chmod +x scripts/setup.sh
./scripts/setup.sh
```

## Alternative Installation Methods

### Google Colab

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Create new notebook
3. Enable GPU: Runtime → Change runtime type → GPU
4. Clone repository and install dependencies

### Cloud GPU

Follow cloud-specific setup guides for AWS, GCP, or Azure.

### Docker

```bash
docker build -t llm-fine-tuning .
docker-compose up jupyter
```

## Verify Installation

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
```

## Next Steps

After successful installation:

1. Read the [Quick Start](quickstart.md) guide
2. Start with [Chapter 1: From Generalist to Specialist](../labs/chapter-01-generalist-to-specialist.md)

## Additional Resources

- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
- [CUDA Installation Guide](https://developer.nvidia.com/cuda-downloads)
- [Hugging Face Documentation](https://huggingface.co/docs)