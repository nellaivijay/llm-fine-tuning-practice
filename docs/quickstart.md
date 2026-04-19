---
title: Quick Start
description: Get started with LLM Fine-Tuning Practice in 5 minutes
---

# Quick Start

Get started with LLM Fine-Tuning Practice in 5 minutes!

## Prerequisites Check

Before starting, ensure you have:

- Python 3.8+ installed
- CUDA-capable GPU (8GB+ VRAM recommended)
- At least 16GB RAM available
- 50GB free disk space

## Quick Setup

### 1. Clone and Setup

```bash
git clone https://github.com/nellaivijay/llm-fine-tuning-practice.git
cd llm-fine-tuning-practice
./scripts/setup.sh
```

### 2. Verify GPU Access

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 3. Start Learning

Open Jupyter Notebook and navigate to the `notebooks/` directory. Start with:

```
notebooks/chapter-01-generalist-to-specialist.ipynb
```

## Your First Fine-Tuning Experiment

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

# Load a small model
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Configure LoRA
lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["c_attn"])
model = get_peft_model(model, lora_config)

# Check trainable parameters
model.print_trainable_parameters()
```

## Chapter Structure

| Chapter | Topic | Duration | Difficulty |
|---------|-------|----------|------------|
| 1 | From Generalist to Specialist | 60-90 min | Beginner |
| 2 | Model Architectures | 60-90 min | Beginner |
| 3 | The Fine-Tuner's Workshop | 60-90 min | Beginner |
| 4 | Data Preparation | 90-120 min | Intermediate |
| 5 | LoRA & PEFT | 90-120 min | Intermediate |
| 6 | The Training Loop | 90-120 min | Intermediate |
| 7 | Evaluation & Alignment | 90-120 min | Advanced |
| 8 | Case Study | 90-120 min | Advanced |
| 9 | Vision-Language Models | 90-120 min | Advanced |

## Learning Path

### For Beginners

1. Start with Chapter 1: From Generalist to Specialist
2. Complete Chapter 3: The Fine-Tuner's Workshop
3. Learn data preparation in Chapter 4

### For Intermediate Users

1. Complete beginner chapters
2. Learn LoRA in Chapter 5
3. Master training loops in Chapter 6

### For Advanced Users

1. Complete all previous chapters
2. Learn evaluation in Chapter 7
3. Apply skills in Chapter 8 Case Study
4. Explore VLMs in Chapter 9

## Tips for Success

1. Understand concepts before running code
2. Start small with smaller models
3. Monitor GPU memory usage
4. Experiment with hyperparameters
5. Use monitoring tools (WandB, TensorBoard)

## Cost Considerations

### Local Training
- Hardware: $500-2000 (one-time)
- Electricity: $20-50/month
- No recurring cloud costs

### Cloud Training
- AWS/GCP: $0.50-3.00/hour
- Colab Pro: $10/month
- Free tiers available

## Next Steps

1. Complete the [Installation Guide](installation.md) if you haven't already
2. Start with [Chapter 1: From Generalist to Specialist](../labs/chapter-01-generalist-to-specialist.md)

## Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [PEFT Library](https://huggingface.co/docs/peft)