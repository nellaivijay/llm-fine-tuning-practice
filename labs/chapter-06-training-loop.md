# Chapter 6: The Training Loop

## Learning Objectives

By the end of this chapter, you will be able to:
- Implement complete training loops for fine-tuning
- Configure learning rate schedules
- Use gradient accumulation for effective batch sizes
- Implement mixed precision training
- Manage checkpoints effectively
- Monitor training progress and metrics

## Duration

90-120 minutes

## Prerequisites

- Chapter 1: From Generalist to Specialist
- Chapter 2: Model Architectures
- Chapter 3: The Fine-Tuner's Workshop
- Chapter 4: Data Preparation
- Chapter 5: LoRA & Parameter-Efficient Fine-Tuning
- Understanding of gradient descent

## Introduction

The training loop is where fine-tuning happens. This chapter covers implementing efficient training loops, optimization techniques, and monitoring strategies.

## Training Loop Components

### Core Components

**1. Data Loading**
- Batch creation
- Data preprocessing
- Collation functions
- Data augmentation

**2. Forward Pass**
- Model inference
- Loss computation
- Metric calculation
- Output generation

**3. Backward Pass**
- Gradient computation
- Gradient clipping
- Optimizer step
- Learning rate scheduling

**4. Monitoring**
- Loss tracking
- Metric logging
- Progress visualization
- Checkpoint saving

## Data Loading and Preprocessing

### Dataset Preparation

**Create Dataset**:
```python
from datasets import Dataset
from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
tokenizer.pad_token = tokenizer.eos_token

# Prepare dataset
def preprocess_function(examples):
    # Tokenize inputs
    model_inputs = tokenizer(
        examples["input"],
        max_length=512,
        truncation=True,
        padding="max_length"
    )

    # Tokenize targets
    labels = tokenizer(
        examples["output"],
        max_length=512,
        truncation=True,
        padding="max_length"
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Apply preprocessing
tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset.column_names
)
```

### Data Collation

**Custom Collator**:
```python
from transformers import DataCollatorForSeq2Seq

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,
    return_tensors="pt"
)
```

**Custom Collator for Instruction Tuning**:
```python
def custom_data_collator(features):
    batch = {}
    for key in features[0].keys():
        batch[key] = torch.stack([f[key] for f in features])
    return batch
```

## Training Arguments Configuration

### Basic Training Arguments

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./outputs",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    warmup_steps=100,
    logging_steps=10,
    save_steps=500,
    eval_steps=500,
    evaluation_strategy="steps",
    save_strategy="steps",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="wandb",  # or "tensorboard", "mlflow"
    run_name="my-finetuning-run",
    fp16=True,  # Mixed precision
)
```

### Advanced Training Arguments

```python
training_args = TrainingArguments(
    output_dir="./outputs",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    weight_decay=0.01,
    warmup_ratio=0.1,  # 10% of steps for warmup
    lr_scheduler_type="cosine",
    logging_steps=10,
    save_steps=500,
    eval_steps=500,
    evaluation_strategy="steps",
    save_strategy="steps",
    save_total_limit=3,  # Keep only 3 best checkpoints
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    fp16=True,
    gradient_checkpointing=True,
    optim="adamw_torch",
    max_grad_norm=1.0,
    ddp_find_unused_parameters=False,
    report_to="wandb",
    run_name="my-finetuning-run",
    logging_dir="./logs",
    seed=42,
    data_seed=42,
)
```

## Learning Rate Scheduling

### Learning Rate Schedulers

**1. Linear Schedule**:
```python
training_args = TrainingArguments(
    lr_scheduler_type="linear",
    learning_rate=2e-4,
    warmup_steps=100,
)
```

**2. Cosine Schedule** (recommended):
```python
training_args = TrainingArguments(
    lr_scheduler_type="cosine",
    learning_rate=2e-4,
    warmup_ratio=0.1,
)
```

**3. Cosine with Restarts**:
```python
training_args = TrainingArguments(
    lr_scheduler_type="cosine_with_restarts",
    learning_rate=2e-4,
    warmup_steps=100,
)
```

**4. Polynomial Schedule**:
```python
training_args = TrainingArguments(
    lr_scheduler_type="polynomial",
    learning_rate=2e-4,
    warmup_steps=100,
)
```

### Custom Learning Rate Schedule

```python
from transformers import get_cosine_schedule_with_warmup

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=100,
    num_training_steps=total_steps
)
```

## Gradient Accumulation

### Why Gradient Accumulation?

**Effective Batch Size**:
```
effective_batch_size = per_device_batch_size × gradient_accumulation_steps × num_gpus
```

**Example**:
```python
# GPU can only fit batch_size=4
# Want effective batch_size=64
# Use gradient_accumulation_steps=16

training_args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=16,
)
```

### Gradient Accumulation Implementation

**Manual Implementation**:
```python
accumulation_steps = 4
optimizer.zero_grad()

for i, batch in enumerate(train_dataloader):
    outputs = model(**batch)
    loss = outputs.loss / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**With Trainer** (automatic):
```python
training_args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
)
```

## Mixed Precision Training

### Benefits

**Memory Reduction**:
- FP16: 50% memory reduction
- BF16: Better numerical stability

**Speed Improvement**:
- Faster tensor operations
- Better GPU utilization

### Implementation

**FP16 Training**:
```python
training_args = TrainingArguments(
    fp16=True,
)
```

**BF16 Training** (if supported):
```python
training_args = TrainingArguments(
    bf16=True,
)
```

**Automatic Detection**:
```python
training_args = TrainingArguments(
    fp16=torch.cuda.is_available(),
    bf16=torch.cuda.is_bf16_supported(),
)
```

### Gradient Checkpointing

**Memory Optimization**:
```python
training_args = TrainingArguments(
    gradient_checkpointing=True,
)
```

**Trade-offs**:
- Saves memory (20-30% reduction)
- Slows training (~20% overhead)
- Good for large models

## Checkpoint Management

### Saving Checkpoints

**Automatic Saving**:
```python
training_args = TrainingArguments(
    save_strategy="steps",
    save_steps=500,
    save_total_limit=3,  # Keep only 3 checkpoints
)
```

**Manual Saving**:
```python
# Save adapter weights
model.save_pretrained("./checkpoint-500")

# Save full model
model.save_pretrained("./full-model")
```

**Loading Checkpoints**:
```python
# Load adapter
from peft import PeftModel
base_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
model = PeftModel.from_pretrained(base_model, "./checkpoint-500")

# Resume training
trainer = Trainer(model=model, args=training_args, ...)
trainer.train(resume_from_checkpoint="./checkpoint-500")
```

### Checkpoint Selection

**Best Model**:
```python
training_args = TrainingArguments(
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)
```

**Last Model**:
```python
training_args = TrainingArguments(
    load_best_model_at_end=False,
)
```

## Monitoring and Logging

### WandB Integration

```python
import wandb

wandb.init(
    project="llm-fine-tuning",
    name="my-experiment",
    config={
        "model": "mistral-7b",
        "learning_rate": 2e-4,
        "batch_size": 4,
    }
)

training_args = TrainingArguments(
    report_to="wandb",
    run_name="my-experiment",
)
```

### TensorBoard Integration

```python
training_args = TrainingArguments(
    report_to="tensorboard",
    logging_dir="./logs",
)

# View with tensorboard
# tensorboard --logdir ./logs
```

### Custom Metrics

```python
import numpy as np
from sklearn.metrics import accuracy_score

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=-1)

    accuracy = accuracy_score(labels.flatten(), predictions.flatten())

    return {"accuracy": accuracy}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)
```

## Complete Training Example

### Using Hugging Face Trainer

```python
from transformers import Trainer, TrainingArguments

# Configure training
training_args = TrainingArguments(
    output_dir="./outputs",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    weight_decay=0.01,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    logging_steps=10,
    save_steps=500,
    eval_steps=500,
    evaluation_strategy="steps",
    save_strategy="steps",
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    fp16=True,
    gradient_checkpointing=True,
    report_to="wandb",
    run_name="mistral-7b-finetune",
)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# Train
trainer.train()

# Save final model
trainer.save_model("./final-model")
```

### Custom Training Loop

```python
import torch
from tqdm import tqdm

# Setup
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=100,
    num_training_steps=total_steps
)

# Training loop
model.train()
accumulation_steps = 4

for epoch in range(num_epochs):
    optimizer.zero_grad()
    
    for step, batch in enumerate(tqdm(train_dataloader)):
        # Move to GPU
        batch = {k: v.to(model.device) for k, v in batch.items()}
        
        # Forward pass
        outputs = model(**batch)
        loss = outputs.loss / accumulation_steps
        
        # Backward pass
        loss.backward()
        
        # Gradient accumulation
        if (step + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        # Logging
        if step % 10 == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}")
        
        # Save checkpoint
        if step % 500 == 0:
            model.save_pretrained(f"./checkpoint-step-{step}")
    
    # Evaluation
    model.eval()
    eval_loss = 0
    with torch.no_grad():
        for batch in eval_dataloader:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            outputs = model(**batch)
            eval_loss += outputs.loss.item()
    
    print(f"Epoch {epoch}, Eval Loss: {eval_loss / len(eval_dataloader):.4f}")
    model.train()
```

## Training Optimization Techniques

### Gradient Clipping

```python
training_args = TrainingArguments(
    max_grad_norm=1.0,
)
```

### Weight Decay

```python
training_args = TrainingArguments(
    weight_decay=0.01,
)
```

### Optimizer Selection

```python
training_args = TrainingArguments(
    optim="adamw_torch",  # or "adamw_hf", "adafactor"
)
```

### Early Stopping

```python
from transformers import EarlyStoppingCallback

trainer = Trainer(
    model=model,
    args=training_args,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)
```

## Practical Exercise

### Exercise 1: Training Configuration

Configure training for a 7B model with:
- Effective batch size: 32
- GPU memory: 12GB
- Dataset size: 10000 examples
- Training epochs: 3

**Calculate**:
1. Per-device batch size
2. Gradient accumulation steps
3. Total training steps
4. Warmup steps (10% of total)

### Exercise 2: Learning Rate Schedule

Compare different learning rate schedules:
1. Linear with 100 warmup steps
2. Cosine with 10% warmup ratio
3. Polynomial with 100 warmup steps

Plot and compare the learning rate curves.

### Exercise 3: Monitoring Setup

Set up monitoring with:
1. WandB integration
2. Custom metrics (accuracy, perplexity)
3. Checkpoint every 500 steps
4. Keep only 3 best checkpoints

## Key Takeaways

1. **Training loops** require careful configuration
2. **Learning rate scheduling** is crucial for convergence
3. **Gradient accumulation** enables larger effective batch sizes
4. **Mixed precision** reduces memory and speeds up training
5. **Checkpoint management** saves progress and best models
6. **Monitoring** provides insights into training progress
7. **Optimization techniques** improve training stability

## Next Steps

With training complete, let's evaluate and align our models in [Chapter 7: Evaluation & Alignment](chapter-07-evaluation-alignment.md).

## Additional Resources

- [Hugging Face Trainer Docs](https://huggingface.co/docs/transformers/training)
- [Learning Rate Schedulers](https://huggingface.co/docs/transformers/main/en/main_classes/optimizer_schedules)
- [WandB Documentation](https://docs.wandb.ai/)