# Chapter 3: The Fine-Tuner's Workshop

## Learning Objectives

By the end of this chapter, you will be able to:
- Set up a complete fine-tuning development environment
- Configure GPU vs. CPU vs. cloud training options
- Install and configure essential dependencies
- Verify your environment is ready for training
- Run your first fine-tuning experiment

## Duration

60-90 minutes

## Prerequisites

- Chapter 1: From Generalist to Specialist
- Chapter 2: Model Architectures
- Python 3.8+ installed
- Basic command line familiarity

## Introduction

This chapter guides you through setting up a complete fine-tuning environment. We'll cover local GPU setup, cloud options, and ensure everything is configured correctly before training.

## Environment Options

### Option 1: Local GPU (Recommended for Learning)

**Advantages**:
- No recurring cloud costs
- Full control over environment
- Can train anytime
- Better for experimentation

**Disadvantages**:
- Upfront hardware cost
- Limited to one GPU
- Maintenance required

**Recommended Hardware**:
- **Minimum**: RTX 3060 (12GB) for 7B models with QLoRA
- **Recommended**: RTX 3090/4090 (24GB) for 13B models
- **Professional**: A100 (40GB/80GB) for full fine-tuning

### Option 2: Google Colab (Free/Cheap)

**Advantages**:
- Free tier available
- No hardware setup
- Easy to share
- Good for learning

**Disadvantages**:
- Limited free GPU time
- Session timeouts
- Less control
- Data privacy concerns

**Setup**:
1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Create new notebook
3. Enable GPU: Runtime → Change runtime type → GPU
4. Install dependencies in notebook cells

### Option 3: Cloud GPU (AWS/GCP/Azure)

**Advantages**:
- Access to powerful GPUs
- Scalable resources
- No hardware maintenance
- Professional infrastructure

**Disadvantages**:
- Recurring costs
- Setup complexity
- Data transfer costs
- Learning curve

**Popular Cloud Options**:
- **AWS**: p3/p4 instances, EC2
- **GCP**: A100 instances, Compute Engine
- **Azure**: ND series, Virtual Machines

## Local GPU Setup

### Step 1: Install NVIDIA Drivers

**Linux**:
```bash
# Check NVIDIA driver version
nvidia-smi

# If not installed:
sudo apt-get update
sudo apt-get install nvidia-driver-535
```

**Windows**:
1. Download from [NVIDIA website](https://www.nvidia.com/Download/index.aspx)
2. Install and restart

**Verify**:
```bash
nvidia-smi
```

### Step 2: Install CUDA

**Linux**:
```bash
# Download CUDA 11.8 or 12.1
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_linux.run
sudo sh cuda_11.8.0_linux.run

# Add to PATH
echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

**Verify**:
```bash
nvcc --version
```

### Step 3: Install Python Dependencies

**Create Virtual Environment**:
```bash
python3 -m venv llm-env
source llm-env/bin/activate
```

**Install PyTorch with CUDA**:
```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Install Project Dependencies**:
```bash
cd llm-fine-tuning-practice
pip install -r requirements.txt
```

### Step 4: Verify GPU Access

**Test PyTorch GPU**:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU count: {torch.cuda.device_count()}")
print(f"GPU name: {torch.cuda.get_device_name(0)}")
```

**Expected Output**:
```
CUDA available: True
CUDA version: 11.8
GPU count: 1
GPU name: NVIDIA GeForce RTX 3090
```

## Docker Setup (Optional)

### Build Docker Image

```bash
cd llm-fine-tuning-practice
docker build -t llm-fine-tuning .
```

### Run Jupyter with GPU

```bash
docker-compose up jupyter
```

### Access Jupyter

Open browser to: http://localhost:8888

## Cloud Setup Guide

### AWS Setup

**1. Create EC2 Instance**:
- Choose region with GPU availability
- Select instance type: p3.2xlarge (V100) or p4d.24xlarge (A100)
- Use Deep Learning AMI

**2. Configure Security Group**:
- Allow SSH (port 22)
- Allow Jupyter (port 8888)

**3. Connect and Setup**:
```bash
ssh -i key.pem ubuntu@instance-ip
# Follow local setup steps
```

**4. Cost Management**:
- Use spot instances for savings
- Stop instances when not in use
- Set up billing alerts

### GCP Setup

**1. Create VM Instance**:
- Go to Compute Engine
- Select region with GPU availability
- Choose machine type with GPU (A100)
- Use Deep Learning VM Image

**2. Install Dependencies**:
```bash
# Similar to local setup
sudo apt-get update
# Install CUDA and dependencies
```

**3. SSH and Setup**:
```bash
gcloud compute ssh instance-name
```

## Essential Dependencies

### Core Libraries

**PyTorch**:
- Deep learning framework
- GPU acceleration
- Automatic differentiation

**Transformers**:
- Hugging Face library
- Pre-trained models
- Tokenizers

**PEFT**:
- Parameter-Efficient Fine-Tuning
- LoRA, QLoRA implementations
- Memory-efficient training

**bitsandbytes**:
- Quantization support
- 4-bit/8-bit quantization
- Memory optimization

**Accelerate**:
- Distributed training
- Mixed precision
- Gradient accumulation

### Training Utilities

**Weights & Biases (WandB)**:
- Experiment tracking
- Metrics visualization
- Hyperparameter optimization

**MLflow**:
- Model registry
- Experiment tracking
- Deployment tools

**TensorBoard**:
- Training visualization
- Loss curves
- Metrics tracking

## Configuration

### Environment Variables

**Create .env file**:
```bash
cp .env.example .env
```

**Edit .env with your settings**:
```bash
# GPU Configuration
CUDA_VISIBLE_DEVICES=0
MAX_MEMORY_GB=24

# Model Configuration
BASE_MODEL=mistralai/Mistral-7B-v0.1

# Training Configuration
BATCH_SIZE=4
LEARNING_RATE=2e-4
NUM_EPOCHS=3

# LoRA Configuration
LORA_R=8
LORA_ALPHA=16
```

### Hugging Face Authentication

**1. Create Hugging Face Account**:
- Go to [huggingface.co](https://huggingface.co)
- Sign up for free account

**2. Generate Access Token**:
- Profile → Settings → Access Tokens
- Create new token
- Copy token

**3. Login**:
```bash
huggingface-cli login
# Paste token when prompted
```

## First Fine-Tuning Experiment

### Quick Test

**Create test script**:
```python
# test_setup.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("Testing GPU access...")
assert torch.cuda.is_available(), "CUDA not available"
print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")

print("\nTesting model loading...")
model_name = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
print(f"✓ Tokenizer loaded")

print("\nTesting PEFT...")
from peft import LoraConfig, get_peft_model
config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"])
print(f"✓ PEFT configured")

print("\n✅ All tests passed! Environment ready for fine-tuning.")
```

**Run test**:
```bash
python test_setup.py
```

### Memory Test

**Check GPU memory**:
```python
import torch
print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
print(f"Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
```

## Common Setup Issues

### CUDA Version Mismatch

**Problem**: PyTorch CUDA version doesn't match system CUDA

**Solution**:
```bash
# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Out of Memory

**Problem**: Training runs out of GPU memory

**Solutions**:
- Reduce batch size
- Use gradient accumulation
- Enable QLoRA (4-bit quantization)
- Reduce sequence length

### Permission Errors

**Problem**: Cannot write to directories

**Solution**:
```bash
chmod -R 755 data models outputs checkpoints
```

## Practical Exercise

### Exercise 1: Environment Setup

1. Set up your preferred environment (local GPU, Colab, or cloud)
2. Install all dependencies
3. Verify GPU access
4. Run the test script
5. Document any issues and solutions

### Exercise 2: Memory Estimation

Given your GPU, estimate:
1. Maximum model size for full fine-tuning
2. Maximum model size for LoRA
3. Maximum model size for QLoRA
4. Which method should you use for a 7B model?

## Key Takeaways

1. **Multiple environment options** - choose based on budget and needs
2. **Local GPU** is best for learning and experimentation
3. **Cloud GPUs** provide access to powerful hardware
4. **Colab** is great for free learning
5. **Proper setup** is crucial for successful training
6. **Test your environment** before starting training
7. **Memory management** is key for GPU training

## Next Steps

With your environment ready, let's prepare data in [Chapter 4: Data Preparation](chapter-04-data-preparation.md).

## Additional Resources

- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
- [CUDA Installation Guide](https://developer.nvidia.com/cuda-downloads)
- [Hugging Face Documentation](https://huggingface.co/docs)
- [PEFT Documentation](https://huggingface.co/docs/peft)