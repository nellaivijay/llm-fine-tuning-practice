#!/bin/bash

# Training Script for LLM Fine-Tuning Practice
# This script runs fine-tuning with specified parameters

set -e

# Load environment variables
if [ -f ".env" ]; then
    export $(cat .env | grep -v '^#' | xargs)
else
    echo "Warning: .env file not found"
fi

# Default values
MODEL_NAME=${BASE_MODEL:-"mistralai/Mistral-7B-v0.1"}
DATA_DIR=${DATA_DIR:-"./data"}
OUTPUT_DIR=${OUTPUT_DIR:-"./outputs"}
BATCH_SIZE=${BATCH_SIZE:-4}
LEARNING_RATE=${LEARNING_RATE:-2e-4}
NUM_EPOCHS=${NUM_EPOCHS:-3}
USE_QLoRA=${USE_QLoRA:-"true"}

echo "======================================"
echo "LLM Fine-Tuning Training"
echo "======================================"
echo "Model: $MODEL_NAME"
echo "Data directory: $DATA_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Batch size: $BATCH_SIZE"
echo "Learning rate: $LEARNING_RATE"
echo "Epochs: $NUM_EPOCHS"
echo "Use QLoRA: $USE_QLoRA"
echo "======================================"

# Activate virtual environment
if [ -d "llm-env" ]; then
    source llm-env/bin/activate
else
    echo "Error: Virtual environment not found. Run setup.sh first."
    exit 1
fi

# Run training
python3 -c "
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model
from datasets import load_from_disk

print('Loading model...')
if '$USE_QLoRA' == 'true':
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        '$MODEL_NAME',
        quantization_config=bnb_config,
        device_map='auto'
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        '$MODEL_NAME',
        torch_dtype=torch.float16,
        device_map='auto'
    )

print('Loading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained('$MODEL_NAME')
tokenizer.pad_token = tokenizer.eos_token

print('Applying LoRA...')
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=['q_proj', 'v_proj'],
    lora_dropout=0.05,
    bias='none',
    task_type='CAUSAL_LM'
)
model = get_peft_model(model, lora_config)

print('Configuring training...')
training_args = TrainingArguments(
    output_dir='$OUTPUT_DIR',
    num_train_epochs=$NUM_EPOCHS,
    per_device_train_batch_size=$BATCH_SIZE,
    per_device_eval_batch_size=$BATCH_SIZE,
    gradient_accumulation_steps=4,
    learning_rate=$LEARNING_RATE,
    weight_decay=0.01,
    warmup_ratio=0.1,
    lr_scheduler_type='cosine',
    logging_steps=10,
    save_steps=500,
    fp16=True,
    gradient_checkpointing=True,
)

print('Loading dataset...')
try:
    train_dataset = load_from_disk('$DATA_DIR/train')
    eval_dataset = load_from_disk('$DATA_DIR/eval')
except:
    print('Dataset not found. Please prepare data first.')
    exit(1)

print('Starting training...')
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()

print('Saving model...')
trainer.save_model('$OUTPUT_DIR/final')
tokenizer.save_pretrained('$OUTPUT_DIR/final')

print('Training completed successfully!')
"

echo ""
echo "Training completed!"
echo "Model saved to: $OUTPUT_DIR/final"