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

That's it! The setup script will:
- Download and configure dependencies
- Create necessary directories
- Verify GPU access
- Display access information

### 2. Verify GPU Access

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

Expected output: `CUDA available: True`

### 3. Start Learning

Open Jupyter Notebook and navigate to the `notebooks/` directory. Start with:

```
notebooks/chapter-01-generalist-to-specialist.ipynb
```

## Your First Fine-Tuning Experiment

### In Jupyter Notebook

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

# Load a small model for testing
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Configure LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["c_attn"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)

# Apply LoRA
model = get_peft_model(model, lora_config)

# Check trainable parameters
model.print_trainable_parameters()

# Generate text
prompt = "Hello, world!"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

## Common Commands

### Start/Stop Services

```bash
# Start training services
./scripts/start.sh

# Stop training services
./scripts/stop.sh

# Check GPU health
./scripts/health-check.sh
```

### View Training Progress

```bash
# View logs
tail -f logs/training.log

# Monitor GPU usage
watch -n 1 nvidia-smi

# Check TensorBoard
tensorboard --logdir logs/
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

### 1. Understand the Concepts

Before running code, read the explanations in the lab markdown files.

### 2. Start Small

- Use smaller models (7B) for learning
- Use QLoRA for memory efficiency
- Test with small datasets first

### 3. Monitor Resources

- Watch GPU memory usage
- Monitor training progress
- Check for out-of-memory errors

### 4. Experiment

Don't just copy the code. Try modifying it:
- Change hyperparameters
- Try different models
- Experiment with data

### 5. Use the Spark UI

Monitor your jobs using available tools:
- WandB for experiment tracking
- TensorBoard for visualization
- nvidia-smi for GPU monitoring

## Cost Considerations

### Local Training

- **Hardware**: One-time GPU purchase ($500-2000)
- **Electricity**: $20-50/month
- **No recurring cloud costs**

### Cloud Training

- **AWS/GCP**: $0.50-3.00/hour for GPU instances
- **Colab Pro**: $10/month for faster GPUs
- **Free Tiers**: Limited but usable

### Cost Optimization

- Use LoRA/QLoRA to reduce memory
- Use smaller base models
- Optimize batch size
- Leverage free tiers

## Next Steps

1. Complete the [Installation Guide](Installation-Guide.md) if you haven't already
2. Start with [Chapter 1: From Generalist to Specialist](../labs/chapter-01-generalist-to-specialist.md)
3. Explore the [Lab Descriptions](../labs/) for detailed information

## Resources

- [Official PyTorch Documentation](https://pytorch.org/docs/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [PEFT Library](https://huggingface.co/docs/peft)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)