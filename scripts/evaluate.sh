#!/bin/bash

# Evaluation Script for LLM Fine-Tuning Practice
# This script evaluates a fine-tuned model

set -e

# Load environment variables
if [ -f ".env" ]; then
    export $(cat .env | grep -v '^#' | xargs)
else
    echo "Warning: .env file not found"
fi

# Default values
MODEL_PATH=${MODEL_PATH:-"./outputs/final"}
DATA_DIR=${DATA_DIR:-"./data"}

echo "======================================"
echo "LLM Fine-Tuning Evaluation"
echo "======================================"
echo "Model path: $MODEL_PATH"
echo "Data directory: $DATA_DIR"
echo "======================================"

# Activate virtual environment
if [ -d "llm-env" ]; then
    source llm-env/bin/activate
else
    echo "Error: Virtual environment not found. Run setup.sh first."
    exit 1
fi

# Run evaluation
python3 -c "
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk
import numpy as np
from sklearn.metrics import accuracy_score

print('Loading model...')
model = AutoModelForCausalLM.from_pretrained(
    '$MODEL_PATH',
    torch_dtype=torch.float16,
    device_map='auto'
)
tokenizer = AutoTokenizer.from_pretrained('$MODEL_PATH')
tokenizer.pad_token = tokenizer.eos_token

print('Loading evaluation dataset...')
try:
    eval_dataset = load_from_disk('$DATA_DIR/eval')
except:
    print('Evaluation dataset not found.')
    exit(1)

print('Running evaluation...')
model.eval()
total_loss = 0
num_samples = 0

with torch.no_grad():
    for i, example in enumerate(eval_dataset):
        if i >= 100:  # Evaluate on first 100 samples
            break
        
        inputs = tokenizer(
            example['input'],
            return_tensors='pt',
            truncation=True,
            max_length=512
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        outputs = model(**inputs, labels=inputs['input_ids'])
        total_loss += outputs.loss.item()
        num_samples += 1
        
        if (i + 1) % 10 == 0:
            print(f'Evaluated {i + 1} samples...')

avg_loss = total_loss / num_samples
perplexity = np.exp(avg_loss)

print(f'Average loss: {avg_loss:.4f}')
print(f'Perplexity: {perplexity:.2f}')

# Generate sample outputs
print()
print('Generating sample outputs...')
for i in range(3):
    prompt = eval_dataset[i]['input']
    inputs = tokenizer(prompt, return_tensors='pt')
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    outputs = model.generate(**inputs, max_length=200)
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print(f'\\nSample {i + 1}:')
    print(f'Prompt: {prompt[:100]}...')
    print(f'Generated: {generated[len(prompt):][:200]}...')

print()
print('Evaluation completed successfully!')
"

echo ""
echo "Evaluation completed!"
echo "Results saved to logs/evaluation.log"