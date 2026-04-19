# Chapter 5: LoRA & Parameter-Efficient Fine-Tuning

## Learning Objectives

By the end of this chapter, you will be able to:
- Understand LoRA (Low-Rank Adaptation) architecture
- Implement QLoRA for memory efficiency
- Use the PEFT library for efficient fine-tuning
- Select appropriate hyperparameters for LoRA
- Compare LoRA vs. full fine-tuning performance

## Duration

90-120 minutes

## Prerequisites

- Chapter 1: From Generalist to Specialist
- Chapter 2: Model Architectures
- Chapter 3: The Fine-Tuner's Workshop
- Chapter 4: Data Preparation
- Basic understanding of linear algebra

## Introduction

Parameter-Efficient Fine-Tuning (PEFT) methods like LoRA and QLoRA make fine-tuning accessible by reducing memory requirements while maintaining performance. This chapter covers these techniques in depth.

## Why Parameter-Efficient Fine-Tuning?

### Full Fine-Tuning Challenges

**Memory Requirements**:
- 7B model: ~14GB GPU memory (FP16)
- 13B model: ~26GB GPU memory (FP16)
- 70B model: ~140GB GPU memory (FP16)

**Storage Requirements**:
- Full model checkpoints: 14-140GB per checkpoint
- Multiple checkpoints multiply storage needs

**Training Time**:
- Full parameter updates
- Slower convergence
- Higher computational cost

### PEFT Advantages

**Memory Efficiency**:
- LoRA: 10-100x less memory
- QLoRA: Additional 4-bit quantization benefits
- Enables fine-tuning on consumer GPUs

**Storage Efficiency**:
- Only adapter weights saved (~1-100MB)
- Base model reused
- Multiple adapters per base model

**Performance**:
- Comparable to full fine-tuning
- Faster convergence
- Better generalization often

**Flexibility**:
- Multiple adapters per base model
- Easy to switch between tasks
- Modular approach

## LoRA (Low-Rank Adaptation)

### Core Concept

LoRA adds trainable low-rank decomposition matrices to existing weights instead of updating all parameters.

**Mathematical Formulation**:
```
W' = W + ΔW = W + BA
```

Where:
- W: Original weight matrix (d × k)
- B: Down-projection matrix (d × r)
- A: Up-projection matrix (r × k)
- r: Rank (r << min(d, k))

**Parameter Reduction**:
- Original: d × k parameters
- LoRA: d × r + r × k = r(d + k) parameters
- Reduction ratio: r(d + k) / (d × k) = r(1/d + 1/k)

### LoRA Architecture

**Implementation Steps**:
1. Freeze original model weights
2. Inject low-rank matrices
3. Train only the low-rank matrices
4. Combine weights during inference

**Target Modules**:
- Attention projections (q_proj, k_proj, v_proj, o_proj)
- Feed-forward networks (up_proj, down_proj, gate_proj)
- Layer norms (usually not targeted)

### LoRA Configuration

**Basic Configuration**:
```python
from peft import LoraConfig

lora_config = LoraConfig(
    r=8,                    # Rank
    lora_alpha=16,          # Scaling factor
    target_modules=["q_proj", "v_proj"],  # Target modules
    lora_dropout=0.05,      # Dropout probability
    bias="none",            # Bias training
    task_type="CAUSAL_LM"   # Task type
)
```

**Hyperparameter Selection**:

**Rank (r)**:
- **r=4**: Very memory efficient, good for simple tasks
- **r=8**: Balanced choice (default)
- **r=16**: More capacity, for complex tasks
- **r=32**: High capacity, similar to full fine-tuning

**Alpha (lora_alpha)**:
- Controls scaling: ΔW = (alpha/r) × BA
- Typical values: 16, 32, 64
- Higher alpha = stronger adaptation
- Rule of thumb: alpha = 2r

**Target Modules**:
```python
# For attention only
target_modules=["q_proj", "v_proj"]

# For full attention
target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]

# For attention + MLP
target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]
```

**Dropout**:
- Prevents overfitting
- Typical values: 0.05-0.1
- Higher for smaller datasets

### LoRA Implementation

**Apply LoRA to Model**:
```python
from transformers import AutoModelForCausalLM
from peft import get_peft_model

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    torch_dtype=torch.float16,
    device_map="auto"
)

# Apply LoRA
model = get_peft_model(model, lora_config)

# Print trainable parameters
model.print_trainable_parameters()
# Output: trainable params: 2,097,152 || all params: 7,242,739,712 || trainable%: 0.029
```

**Trainable Parameter Analysis**:
- Original model: 7.2B parameters
- LoRA adapter: 2M parameters
- Reduction: 99.97% fewer trainable parameters

## QLoRA (Quantized LoRA)

### Core Concept

QLoRA combines LoRA with 4-bit quantization for extreme memory efficiency.

**Key Innovations**:
1. **4-bit NormalFloat (NF4)**: Optimized for normally distributed weights
2. **Double Quantization**: Quantizes quantization constants
3. **Paged Optimizers**: CPU offloading for optimizer states

### QLoRA Configuration

**Enable QLoRA**:
```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,              # Use 4-bit quantization
    bnb_4bit_quant_type="nf4",      # NormalFloat4
    bnb_4bit_compute_dtype=torch.float16,  # Compute in FP16
    bnb_4bit_use_double_quant=True, # Double quantization
)

# Load model with quantization
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    quantization_config=bnb_config,
    device_map="auto"
)
```

**Memory Comparison**:

| Method | 7B Model Memory | 13B Model Memory |
|--------|----------------|------------------|
| FP16 Full | 14GB | 26GB |
| LoRA | 14GB | 26GB |
| QLoRA | 6GB | 10GB |

### QLoRA Implementation

**Complete QLoRA Setup**:
```python
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model

# Quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    quantization_config=bnb_config,
    device_map="auto"
)

# LoRA config
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply LoRA
model = get_peft_model(model, lora_config)
```

## PEFT Library

### Supported Methods

**PEFT Methods**:
- **LoRA**: Low-Rank Adaptation
- **QLoRA**: Quantized LoRA
- **Prefix Tuning**: Trainable prefix tokens
- **P-Tuning**: Learnable prompt embeddings
- **Adapters**: Adapter layers
- **IA3**: Learned feature scaling

### PEFT Workflow

**1. Load Base Model**:
```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    torch_dtype=torch.float16
)
```

**2. Configure PEFT**:
```python
from peft import LoraConfig

config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)
```

**3. Apply PEFT**:
```python
from peft import get_peft_model

model = get_peft_model(model, config)
```

**4. Train**:
```python
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

trainer.train()
```

**5. Save Adapter**:
```python
model.save_pretrained("./my-adapter")
```

**6. Load Adapter**:
```python
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1"
)
model = PeftModel.from_pretrained(
    base_model,
    "./my-adapter"
)
```

## Hyperparameter Selection

### Rank Selection

**Guidelines**:
- **Simple tasks**: r=4-8
- **Complex tasks**: r=16-32
- **Domain adaptation**: r=8-16
- **Style transfer**: r=4-8

**Trade-offs**:
- Higher rank → more capacity → more memory
- Lower rank → less capacity → less memory
- Diminishing returns after r=16

### Alpha Selection

**Guidelines**:
- Default: alpha = 2r
- Range: 8-64
- Higher alpha for stronger adaptation

**Example**:
```python
# Conservative
LoraConfig(r=8, lora_alpha=16)

# Aggressive
LoraConfig(r=8, lora_alpha=32)
```

### Target Module Selection

**Conservative** (memory efficient):
```python
target_modules=["q_proj", "v_proj"]
```

**Balanced** (default):
```python
target_modules=["q_proj", "k_proj", "v_proj"]
```

**Comprehensive** (more capacity):
```python
target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]
```

## LoRA vs. Full Fine-Tuning

### Performance Comparison

**Task Performance**:
- LoRA: 95-99% of full fine-tuning performance
- QLoRA: 90-98% of full fine-tuning performance
- Gap decreases with larger datasets

**Memory Usage**:
- Full: 100% memory
- LoRA: 100% memory (same as full)
- QLoRA: 30-40% memory

**Training Speed**:
- LoRA: Similar or faster (fewer parameters)
- QLoRA: Slower due to quantization overhead

**Storage**:
- Full: 14-140GB per checkpoint
- LoRA: 1-100MB per adapter
- QLoRA: 1-100MB per adapter

### When to Use Each

**Use Full Fine-Tuning**:
- Maximum performance required
- Abundant GPU memory
- Complete model adaptation needed
- Research/experimental purposes

**Use LoRA**:
- Good performance with less storage
- Multiple adapters per base model
- Moderate GPU memory available
- Production deployment

**Use QLoRA**:
- Limited GPU memory
- Consumer GPU training
- Cost-sensitive applications
- Good performance acceptable

## Practical Exercise

### Exercise 1: LoRA Configuration

For a 7B model:
1. Configure LoRA with r=8, alpha=16
2. Calculate trainable parameters
3. Compare with r=16, alpha=32
4. Which configuration would you choose?

### Exercise 2: QLoRA Memory Analysis

Given your GPU:
1. Calculate memory for FP16 full fine-tuning
2. Calculate memory for QLoRA
3. Can you fine-tune a 7B model?
4. Can you fine-tune a 13B model?

### Exercise 3: Hyperparameter Tuning

Experiment with:
1. Different ranks (4, 8, 16, 32)
2. Different alphas (8, 16, 32, 64)
3. Different target modules
4. Document performance and memory usage

## Key Takeaways

1. **LoRA dramatically reduces trainable parameters** while maintaining performance
2. **QLoRA adds quantization** for extreme memory efficiency
3. **PEFT library** provides easy implementation
4. **Hyperparameter selection** balances capacity and efficiency
5. **LoRA vs. full fine-tuning** trade-offs depend on use case
6. **Multiple adapters** can share one base model
7. **Memory efficiency** enables consumer GPU training

## Next Steps

Now that we understand efficient fine-tuning methods, let's implement the training loop in [Chapter 6: The Training Loop](chapter-06-training-loop.md).

## Additional Resources

- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [bitsandbytes Documentation](https://github.com/TimDettmers/bitsandbytes)