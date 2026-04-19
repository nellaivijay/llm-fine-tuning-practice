# Chapter 8: Case Study: Real-World Fine-Tuning

## Learning Objectives

By the end of this chapter, you will be able to:
- Design and implement an end-to-end fine-tuning project
- Adapt models for domain-specific tasks
- Consider production deployment strategies
- Implement model serving and inference
- Set up monitoring and maintenance
- Analyze cost and performance trade-offs

## Duration

90-120 minutes

## Prerequisites

- Chapters 1-7: All previous chapters
- Understanding of software engineering principles
- Familiarity with deployment concepts

## Introduction

This chapter applies all previous concepts in a comprehensive case study. We'll build a domain-specific fine-tuned model from start to finish.

## Case Study: Customer Support Chatbot

### Project Overview

**Objective**: Fine-tune an LLM for customer support in a SaaS company

**Requirements**:
- Handle common support queries
- Provide accurate, helpful responses
- Maintain consistent tone and formatting
- Fast response times (<2 seconds)
- Cost-effective deployment

**Domain**: Software-as-a-Service (SaaS) technical support

### Project Planning

**Phase 1: Data Collection (Week 1)**
- Gather historical support tickets
- Clean and curate data
- Generate synthetic examples
- Create evaluation set

**Phase 2: Model Selection (Week 1-2)**
- Evaluate base models
- Select appropriate architecture
- Configure training setup
- Estimate resources

**Phase 3: Fine-Tuning (Week 2-3)**
- Prepare training data
- Configure training pipeline
- Train model with LoRA
- Evaluate and iterate

**Phase 4: Deployment (Week 3-4)**
- Optimize for inference
- Set up serving infrastructure
- Implement monitoring
- Deploy to production

## Data Collection and Preparation

### Historical Data

**Source**: Customer support tickets from past 2 years

**Data Format**:
```json
{
  "ticket_id": "12345",
  "customer_query": "How do I reset my password?",
  "agent_response": "To reset your password, go to Settings > Security > Change Password...",
  "category": "account",
  "sentiment": "neutral",
  "resolution_time": 120
}
```

**Data Cleaning**:
```python
import pandas as pd

# Load data
df = pd.read_csv('support_tickets.csv')

# Remove sensitive information
import re
def remove_pii(text):
    # Remove emails
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
    # Remove phone numbers
    text = re.sub(r'\b\d{3}-\d{3}-\d{4}\b', '[PHONE]', text)
    return text

df['customer_query'] = df['customer_query'].apply(remove_pii)
df['agent_response'] = df['agent_response'].apply(remove_pii)

# Filter for quality
df = df[df['resolution_time'] < 600]  # Under 10 minutes
df = df[df['sentiment'] != 'negative']  # Good resolutions
df = df[df['agent_response'].str.len() > 50]  # Substantial responses
```

### Data Formatting

**Instruction Tuning Format**:
```python
def format_instruction(row):
    return {
        "instruction": f"You are a helpful customer support agent for a SaaS company. Answer the following query.",
        "input": row['customer_query'],
        "output": row['agent_response']
    }

formatted_data = df.apply(format_instruction, axis=1).tolist()
```

**Train/Test Split**:
```python
from sklearn.model_selection import train_test_split

train_data, test_data = train_test_split(
    formatted_data,
    test_size=0.1,
    random_state=42
)

print(f"Training examples: {len(train_data)}")
print(f"Test examples: {len(test_data)}")
```

### Synthetic Data Generation

**Template-Based Generation**:
```python
templates = [
    "How do I {action}?",
    "I'm having trouble with {feature}. What should I do?",
    "Can you help me with {action}?",
    "What's the best way to {action}?",
]

actions = ["reset my password", "cancel my subscription", "upgrade my plan", "export my data"]

synthetic_queries = []
for template in templates:
    for action in actions:
        synthetic_queries.append(template.format(action=action))

# Generate responses using GPT-4
from openai import OpenAI
client = OpenAI(api_key="your-api-key")

synthetic_data = []
for query in synthetic_queries[:50]:  # Limit for cost
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful customer support agent."},
            {"role": "user", "content": query}
        ]
    )
    synthetic_data.append({
        "instruction": "You are a helpful customer support agent.",
        "input": query,
        "output": response.choices[0].message.content
    })

# Combine with real data
combined_data = train_data + synthetic_data
```

## Model Selection

### Base Model Evaluation

**Candidate Models**:
1. Mistral 7B - Efficient, strong performance
2. LLaMA 2 7B - Popular, good documentation
3. CodeLLaMA 7B - If technical support needed

**Selection Criteria**:
- Performance on support tasks
- Inference speed
- Memory requirements
- License compatibility

**Evaluation Script**:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def evaluate_model(model_name, test_queries):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    results = []
    for query in test_queries[:10]:  # Sample
        inputs = tokenizer(query, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=200)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        results.append(response)
    
    # Manual evaluation of quality
    return results

# Evaluate candidates
for model_name in ["mistralai/Mistral-7B-v0.1", "meta-llama/Llama-2-7b-hf"]:
    print(f"\nEvaluating {model_name}")
    results = evaluate_model(model_name, test_queries)
```

**Selection**: Mistral 7B (Apache 2.0 license, good performance, efficient)

## Fine-Tuning Implementation

### Training Configuration

```python
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model

# Load model with QLoRA
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    quantization_config=bnb_config,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
tokenizer.pad_token = tokenizer.eos_token

# LoRA configuration
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
```

### Training Pipeline

```python
# Prepare dataset
from datasets import Dataset

train_dataset = Dataset.from_list(combined_data)
test_dataset = Dataset.from_list(test_data)

def preprocess_function(examples):
    inputs = [
        f"### Instruction:\n{inst}\n\n### Input:\n{inp}\n\n### Response:\n"
        for inst, inp in zip(examples["instruction"], examples["input"])
    ]
    targets = examples["output"]
    
    model_inputs = tokenizer(
        inputs,
        max_length=512,
        truncation=True,
        padding="max_length"
    )
    
    labels = tokenizer(
        targets,
        max_length=512,
        truncation=True,
        padding="max_length"
    )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_train = train_dataset.map(preprocess_function, batched=True)
tokenized_test = test_dataset.map(preprocess_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./support-chatbot",
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
    run_name="support-chatbot",
)

# Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
)

trainer.train()

# Save model
trainer.save_model("./support-chatbot-final")
```

## Evaluation

### Automated Evaluation

```python
# Evaluate on test set
eval_results = trainer.evaluate()
print(f"Test Loss: {eval_results['eval_loss']:.4f}")

# Generate sample responses
def generate_response(query):
    prompt = f"### Instruction:\nYou are a helpful customer support agent.\n\n### Input:\n{query}\n\n### Response:\n"
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=300)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("### Response:\n")[-1]

# Test queries
test_queries = [
    "How do I reset my password?",
    "Can I cancel my subscription?",
    "How do I upgrade my plan?",
]

for query in test_queries:
    response = generate_response(query)
    print(f"Query: {query}")
    print(f"Response: {response}\n")
```

### Human Evaluation

**Create Evaluation Set**:
```python
evaluation_set = test_data[:50]  # 50 examples for human review

# Generate responses
for example in evaluation_set:
    example["generated"] = generate_response(example["input"])

# Save for human evaluation
import json
with open("evaluation_set.json", "w") as f:
    json.dump(evaluation_set, f, indent=2)
```

**Evaluation Criteria**:
1. Accuracy: Is the information correct?
2. Helpfulness: Does it solve the user's problem?
3. Tone: Is it professional and friendly?
4. Clarity: Is the response easy to understand?
5. Completeness: Does it fully address the query?

## Deployment

### Model Optimization

**Quantization for Inference**:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "./support-chatbot-final",
    torch_dtype=torch.float16,
    device_map="auto"
)

# Quantize to 8-bit
from transformers import BitsAndBytesConfig
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
)
model = AutoModelForCausalLM.from_pretrained(
    "./support-chatbot-final",
    quantization_config=bnb_config,
    device_map="auto"
)
```

**Optimization Techniques**:
```python
# Enable Flash Attention (if available)
model.config.use_flash_attention_2 = True

# Optimize for inference
from optimum.bettertransformer import BetterTransformer
model = BetterTransformer.transform(model)
```

### Serving Options

**Option 1: FastAPI Server**:
```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Query(BaseModel):
    text: str

@app.post("/generate")
async def generate(query: Query):
    response = generate_response(query.text)
    return {"response": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**Option 2: Hugging Face Inference API**:
```python
# Upload to Hugging Face
model.push_to_hub("your-username/support-chatbot")
tokenizer.push_to_hub("your-username/support-chatbot")

# Use Inference API
from huggingface_hub import InferenceClient

client = InferenceClient("your-username/support-chatbot")
response = client.text_generation(query, max_new_tokens=200)
```

**Option 3: vLLM for High Performance**:
```bash
pip install vllm

python -m vllm.entrypoints.api_server \
    --model ./support-chatbot-final \
    --port 8000 \
    --quantization awq
```

### Deployment Architecture

```
┌─────────────────┐
│   Load Balancer │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
┌───▼───┐ ┌──▼────┐
│ GPU 1 │ │ GPU 2 │
│ vLLM  │ │ vLLM  │
└───────┘ └───────┘
```

## Monitoring

### Performance Monitoring

**Response Time**:
```python
import time

def generate_response_with_metrics(query):
    start_time = time.time()
    response = generate_response(query)
    end_time = time.time()
    
    metrics = {
        "response_time": end_time - start_time,
        "query_length": len(query),
        "response_length": len(response),
    }
    
    # Log metrics
    log_metrics(metrics)
    
    return response, metrics
```

**Quality Monitoring**:
```python
# Track user feedback
def log_feedback(query, response, rating):
    # Save to database
    # Analyze trends
    pass
```

### Alerting

**Set up alerts for**:
- High response times (>2 seconds)
- Low satisfaction ratings
- High error rates
- GPU memory issues

## Cost Analysis

### Training Costs

**Hardware**: RTX 3090 (24GB)
- Training time: 6 hours
- Electricity: ~$2
- Total training cost: ~$2

**Alternative (Cloud)**:
- AWS p3.2xlarge (V100): $3.06/hour
- Training time: 6 hours
- Total: $18.36

### Inference Costs

**On-Premises**:
- Hardware: RTX 3090 ($1500 one-time)
- Electricity: $50/month
- Total: $650/year (amortized)

**Cloud**:
- AWS g4dn.xlarge (T4): $0.526/hour
- 1000 requests/day × 365 = 365,000 requests/year
- Assuming 1 second per request: 101 hours/year
- Total: $53/year

### Cost Optimization

**Optimizations**:
1. Use smaller model (7B instead of 13B)
2. Quantization (8-bit or 4-bit)
3. Batch requests
4. Caching common queries
5. Use vLLM for efficiency

## Maintenance

### Model Updates

**Retraining Schedule**:
- Monthly with new data
- When performance degrades
- When new features added

**Continuous Improvement**:
- Collect user feedback
- Monitor performance metrics
- Identify failure cases
- Update training data

### Version Management

**Model Versioning**:
```bash
support-chatbot-v1/
support-chatbot-v2/
support-chatbot-v3/
```

**Rollback Strategy**:
- Keep previous versions
- A/B testing
- Gradual rollout

## Practical Exercise

### Exercise 1: End-to-End Project

Choose your own domain and:
1. Collect or create dataset (100 examples)
2. Select base model
3. Fine-tune with LoRA
4. Evaluate performance
5. Document results

### Exercise 2: Deployment Setup

Set up a simple deployment:
1. Create FastAPI server
2. Test locally
3. Document API endpoints
4. Measure response times

### Exercise 3: Cost Analysis

For your project:
1. Calculate training costs
2. Estimate inference costs
3. Compare on-premises vs. cloud
4. Identify optimization opportunities

## Key Takeaways

1. **End-to-end projects** require careful planning
2. **Data quality** is crucial for success
3. **Model selection** balances performance and resources
4. **Fine-tuning** adapts models to domains
5. **Deployment** requires optimization and monitoring
6. **Cost analysis** informs infrastructure decisions
7. **Maintenance** ensures long-term success

## Next Steps

Explore specialized models in [Chapter 9: Vision-Language Models](chapter-09-vision-language-models.md).

## Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [vLLM Documentation](https://vllm.readthedocs.io/)
- [Hugging Face Inference](https://huggingface.co/docs/api-inference)