# Chapter 7: Evaluation & Alignment

## Learning Objectives

By the end of this chapter, you will be able to:
- Select appropriate evaluation metrics for LLMs
- Implement automated evaluation using benchmarks
- Conduct human evaluation strategies
- Understand RLHF (Reinforcement Learning from Human Feedback)
- Implement DPO (Direct Preference Optimization)
- Evaluate model safety and bias

## Duration

90-120 minutes

## Prerequisites

- Chapter 1: From Generalist to Specialist
- Chapter 2: Model Architectures
- Chapter 3: The Fine-Tuner's Workshop
- Chapter 4: Data Preparation
- Chapter 5: LoRA & Parameter-Efficient Fine-Tuning
- Chapter 6: The Training Loop

## Introduction

Evaluation and alignment are critical for ensuring fine-tuned models perform well and behave appropriately. This chapter covers metrics, benchmarks, human evaluation, and alignment techniques.

## Evaluation Metrics

### Automatic Metrics

**1. Perplexity**
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./my-model")
tokenizer = AutoTokenizer.from_pretrained("./my-model")

def calculate_perplexity(text, model, tokenizer):
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids
    target_ids = input_ids.clone()

    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)
        loss = outputs.loss

    return torch.exp(loss)

perplexity = calculate_perplexity("Hello, world!", model, tokenizer)
print(f"Perplexity: {perplexity:.2f}")
```

**2. BLEU Score** (for translation/generation):
```python
from datasets import load_metric
bleu = load_metric("bleu")

predictions = [["hello", "world"]]
references = [["hello", "world"]]

results = bleu.compute(predictions=predictions, references=references)
print(f"BLEU: {results['bleu']:.4f}")
```

**3. ROUGE Score** (for summarization):
```python
from datasets import load_metric
rouge = load_metric("rouge")

predictions = ["summary text here"]
references = ["reference summary"]

results = rouge.compute(predictions=predictions, references=references)
print(f"ROUGE-1: {results['rouge1']:.4f}")
```

**4. Exact Match** (for QA):
```python
def exact_match(predicted, reference):
    return predicted.strip().lower() == reference.strip().lower()

em_score = exact_match("Paris", "paris")
print(f"Exact Match: {em_score}")
```

### Task-Specific Metrics

**Classification Metrics**:
```python
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_classification_metrics(predictions, labels):
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted'
    )
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
```

**Generation Quality Metrics**:
```python
def compute_generation_metrics(generations, references):
    # Length ratio
    avg_gen_len = np.mean([len(g.split()) for g in generations])
    avg_ref_len = np.mean([len(r.split()) for r in references])
    length_ratio = avg_gen_len / avg_ref_len

    # Distinct-n (diversity)
    def distinct_n(generations, n):
        ngrams = set()
        for gen in generations:
            words = gen.split()
            for i in range(len(words) - n + 1):
                ngrams.add(tuple(words[i:i+n]))
        return len(ngrams) / sum(len(g.split()) - n + 1 for g in generations)

    dist_1 = distinct_n(generations, 1)
    dist_2 = distinct_n(generations, 2)

    return {
        "length_ratio": length_ratio,
        "distinct_1": dist_1,
        "distinct_2": dist_2
    }
```

## Benchmark Evaluation

### Popular Benchmarks

**1. MMLU (Massive Multitask Language Understanding)**:
```python
from datasets import load_dataset

mmlu = load_dataset("cais/mmlu", "all")
```

**2. GSM8K (Grade School Math)**:
```python
gsm8k = load_dataset("gsm8k", "main")
```

**3. HellaSwag (Commonsense Reasoning)**:
```python
hellaswag = load_dataset("hellaswag", "validation")
```

**4. TruthfulQA (Truthfulness)**:
```python
truthfulqa = load_dataset("truthful_qa", "validation")
```

### Running Benchmarks

**Using EleutherAI LM Evaluation Harness**:
```bash
git clone https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .

# Evaluate model
python main.py \
    --model hf \
    --model_args pretrained=./my-model \
    --tasks hellaswag \
    --device cuda
```

**Custom Benchmark Evaluation**:
```python
def evaluate_on_benchmark(model, tokenizer, benchmark, task):
    results = []
    for example in benchmark:
        prompt = example["question"]
        inputs = tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=100)
        
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Compare with reference
        score = compare_prediction(prediction, example["answer"])
        results.append(score)
    
    return np.mean(results)
```

## Human Evaluation

### Evaluation Strategies

**1. Side-by-Side Comparison**:
- Present model outputs side by side
- Human raters choose preferred output
- Collect preferences and calculate win rate

**2. Likert Scale Rating**:
- Rate outputs on 1-5 scale
- Dimensions: quality, relevance, fluency, safety
- Average ratings across dimensions

**3. Binary Classification**:
- Good vs. bad classification
- Useful for quality filtering
- Simple and fast

**4. Ranking**:
- Rank multiple outputs
- Useful for comparing models
- More nuanced than binary

### Human Evaluation Setup

**Create Evaluation Interface**:
```python
# Simple evaluation script
def human_evaluation_interface(model, tokenizer, test_set):
    for i, example in enumerate(test_set):
        prompt = example["prompt"]
        inputs = tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=200)
        
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"\n=== Example {i+1} ===")
        print(f"Prompt: {prompt}")
        print(f"Prediction: {prediction}")
        print(f"Reference: {example['reference']}")
        
        rating = input("Rate 1-5: ")
        # Save rating
```

**Using Crowdsourcing Platforms**:
- Amazon Mechanical Turk
- Scale AI
- Surge AI
- Prolific

### Best Practices

**1. Clear Guidelines**:
- Define evaluation criteria
- Provide examples
- Train raters

**2. Multiple Raters**:
- Use 3-5 raters per item
- Calculate inter-rater agreement
- Aggregate ratings

**3. Randomization**:
- Randomize order of examples
- Blind model identity
- Prevent bias

**4. Quality Control**:
- Attention checks
- Trap questions
- Filter low-quality raters

## RLHF (Reinforcement Learning from Human Feedback)

### RLHF Pipeline

**1. Supervised Fine-Tuning (SFT)**:
- Train on human demonstrations
- Learn basic instruction following
- Foundation for RLHF

**2. Reward Model Training**:
- Collect human preference data
- Train reward model to predict preferences
- Learn human values

**3. Reinforcement Learning**:
- Use reward model as feedback
- Optimize policy with PPO
- Align with human preferences

### Reward Model Training

**Preference Data Collection**:
```python
# Preference data format
preference_data = [
    {
        "prompt": "What is the capital of France?",
        "chosen": "The capital of France is Paris.",
        "rejected": "France's capital is Berlin."
    },
    # ... more examples
]
```

**Train Reward Model**:
```python
from transformers import AutoModelForSequenceClassification

reward_model = AutoModelForSequenceClassification.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    num_labels=1  # Scalar reward
)

# Train on preference pairs
# Loss: -log(sigmoid(chosen_score - rejected_score))
```

### PPO Training

**Using TRL Library**:
```python
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

# Load model with value head
model = AutoModelForCausalLMWithValueHead.from_pretrained(
    "./sft-model"
)

# Configure PPO
ppo_config = PPOConfig(
    batch_size=128,
    mini_batch_size=32,
    learning_rate=1.41e-5,
)

# Create PPO trainer
ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=model,
    ref_model=None,
    tokenizer=tokenizer,
    dataset=dataset,
    reward_model=reward_model,
)

# Train with PPO
ppo_trainer.train()
```

## DPO (Direct Preference Optimization)

### DPO Overview

DPO simplifies RLHF by:
- Eliminating need for separate reward model
- Directly optimizing on preference data
- More stable and simpler to implement

### DPO Implementation

**Using TRL Library**:
```python
from trl import DPOTrainer

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "./sft-model"
)

# Configure DPO
dpo_config = DPOConfig(
    beta=0.1,  # DPO temperature
    learning_rate=5e-7,
)

# Create DPO trainer
dpo_trainer = DPOTrainer(
    model=model,
    ref_model=None,
    args=dpo_config,
    train_dataset=preference_dataset,
    tokenizer=tokenizer,
)

# Train with DPO
dpo_trainer.train()
```

**Preference Data Format**:
```python
preference_dataset = [
    {
        "prompt": "What is machine learning?",
        "chosen": "Machine learning is a subset of AI...",
        "rejected": "ML is when computers learn..."
    },
    # ... more examples
]
```

## Safety and Bias Evaluation

### Safety Evaluation

**1. Toxicity Detection**:
```python
from detoxify import Detoxify

detoxify = Detoxify('original')

def evaluate_toxicity(text):
    results = detoxify.predict(text)
    return results['toxicity']

toxicity_score = evaluate_toxicity("This is a test")
```

**2. Safety Classifiers**:
```python
# Use safety classifiers from OpenAI, Anthropic, etc.
def evaluate_safety(text):
    # Check for harmful content
    # Check for PII
    # Check for bias
    pass
```

### Bias Evaluation

**1. Gender Bias**:
```python
def evaluate_gender_bias(model, tokenizer):
    prompts = [
        "The doctor said he",
        "The nurse said she",
        "The engineer said he",
        "The teacher said she",
    ]
    
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_length=20)
        completion = tokenizer.decode(outputs[0])
        print(f"{prompt} -> {completion}")
```

**2. Cultural Bias**:
```python
def evaluate_cultural_bias(model, tokenizer):
    # Test across different cultural contexts
    # Check for stereotypical associations
    pass
```

**3. Using Bias Benchmarks**:
```python
# CrowS-Pairs bias benchmark
crows_pairs = load_dataset("crows_pairs")
```

## Practical Exercise

### Exercise 1: Metric Implementation

Implement the following metrics:
1. Perplexity calculation
2. BLEU score for generation
3. Exact match for QA
4. Custom task-specific metric

### Exercise 2: Human Evaluation Setup

Create a simple human evaluation interface:
1. Load model and test set
2. Generate predictions
3. Display side-by-side with references
4. Collect ratings
5. Calculate average scores

### Exercise 3: DPO Training

Set up DPO training:
1. Create preference dataset (10 examples)
2. Configure DPO trainer
3. Train on preference data
4. Compare outputs before and after DPO

## Key Takeaways

1. **Multiple evaluation metrics** provide comprehensive assessment
2. **Benchmarks** enable standardized comparison
3. **Human evaluation** is crucial for quality assessment
4. **RLHF** aligns models with human preferences
5. **DPO** simplifies preference optimization
6. **Safety evaluation** ensures responsible deployment
7. **Bias detection** promotes fair models

## Next Steps

Now let's apply everything in a real-world project in [Chapter 8: Case Study](chapter-08-case-study.md).

## Additional Resources

- [Hugging Face Evaluate](https://huggingface.co/docs/evaluate)
- [TRL Library](https://huggingface.co/docs/trl)
- [LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [DPO Paper](https://arxiv.org/abs/2305.18290)