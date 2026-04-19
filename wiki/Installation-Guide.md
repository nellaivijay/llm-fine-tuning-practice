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

# Verify installation
nvidia-smi
```

#### Windows
Download and install from [NVIDIA website](https://www.nvidia.com/Download/index.aspx)

#### macOS
Not supported (use cloud GPUs instead)

### Step 2: Install CUDA

#### Linux
```bash
# Download CUDA 11.8 or 12.1
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_linux.run
sudo sh cuda_11.8.0_linux.run

# Add to PATH
echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify
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
source llm-env/bin/activate  # On Windows: llm-env\Scripts\activate
```

### Step 5: Install Dependencies

```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install project dependencies
pip install -r requirements.txt
```

### Step 6: Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your settings
nano .env  # or use your preferred editor
```

### Step 7: Verify Installation

```bash
# Run setup script
chmod +x scripts/setup.sh
./scripts/setup.sh
```

## Alternative Installation Methods

### Google Colab (Free/Cheap)

**Advantages**:
- Free GPU access
- No setup required
- Easy to use

**Setup**:
1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Create new notebook
3. Enable GPU: Runtime → Change runtime type → GPU
4. Clone repository and install dependencies in notebook cells

### Cloud GPU (AWS/GCP/Azure)

**AWS Setup**:
1. Create EC2 instance with GPU (p3/p4 instances)
2. Use Deep Learning AMI
3. SSH into instance
4. Follow local setup steps

**GCP Setup**:
1. Create VM with GPU (A100 instances)
2. Use Deep Learning VM Image
3. SSH and setup

**Azure Setup**:
1. Create VM with GPU (ND series)
2. Install CUDA and dependencies
3. Follow local setup steps

### Docker Setup

**Build Docker Image**:
```bash
docker build -t llm-fine-tuning .
```

**Run with GPU**:
```bash
docker-compose up jupyter
```

## Verify Installation

### Check GPU Access

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
```

### Test Model Loading

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

print(f"Model loaded successfully")
print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
```

## Troubleshooting

### CUDA Not Found

**Problem**: `CUDA not available` error

**Solution**:
```bash
# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Out of Memory

**Problem**: Training runs out of GPU memory

**Solutions**:
- Use QLoRA (4-bit quantization)
- Reduce batch size
- Use gradient accumulation
- Reduce sequence length

### Permission Errors

**Problem**: Cannot write to directories

**Solution**:
```bash
chmod -R 755 data models outputs checkpoints
```

## Uninstallation

```bash
# Deactivate virtual environment
deactivate

# Remove virtual environment
rm -rf llm-env

# Remove project directory
cd ..
rm -rf llm-fine-tuning-practice
```

## Next Steps

After successful installation:

1. Read the [Quick Start](Quick-Start.md) guide
2. Start with [Chapter 1: From Generalist to Specialist](../labs/chapter-01-generalist-to-specialist.md)
3. Open Jupyter notebooks in `notebooks/` directory

## Additional Resources

- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
- [CUDA Installation Guide](https://developer.nvidia.com/cuda-downloads)
- [Hugging Face Documentation](https://huggingface.co/docs)
- [Google Colab Documentation](https://colab.research.google.com/)