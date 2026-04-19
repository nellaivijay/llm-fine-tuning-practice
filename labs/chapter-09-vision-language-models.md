# Chapter 9: Vision-Language Models

## Learning Objectives

By the end of this chapter, you will be able to:
- Understand vision-language model architectures
- Fine-tune multimodal models for image-text tasks
- Prepare image and text data for VLM training
- Implement VLM fine-tuning with LoRA
- Evaluate vision-language model performance
- Apply VLMs to practical applications

## Duration

90-120 minutes

## Prerequisites

- Chapter 1: From Generalist to Specialist
- Chapter 2: Model Architectures
- Chapter 3: The Fine-Tuner's Workshop
- Chapter 4: Data Preparation
- Chapter 5: LoRA & Parameter-Efficient Fine-Tuning
- Basic understanding of computer vision

## Introduction

Vision-Language Models (VLMs) combine visual and textual understanding, enabling tasks like image captioning, visual question answering, and image-text retrieval. This chapter covers VLM fine-tuning techniques.

## Vision-Language Model Architectures

### Core Concepts

**Vision-Language Models** integrate:
- **Vision Encoder**: Processes images (e.g., CLIP, ViT)
- **Language Model**: Processes text (e.g., GPT, LLaMA)
- **Cross-Attention**: Connects vision and language
- **Alignment**: Learns image-text relationships

### Popular VLM Architectures

**1. CLIP (Contrastive Language-Image Pre-training)**:
- Contrastive learning approach
- Aligns images and text in shared space
- Used for image-text retrieval

**2. BLIP (Bootstrapping Language-Image Pre-training)**:
- Image captioning and VQA
- Decoder-based generation
- Strong multimodal understanding

**3. LLaVA (Large Language-and-Vision Assistant)**:
- Connects vision encoder with LLM
- Instruction-tuned for multimodal chat
- State-of-the-art performance

**4. Flamingo**:
- Few-shot learning for vision-language
- Perceiver-based architecture
- Strong in-context learning

**5. GPT-4V (Vision)**:
- Multimodal capabilities
- Complex visual reasoning
- Proprietary (API access)

### Architecture Components

**Vision Encoder**:
```python
from transformers import CLIPVisionModel

vision_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
```

**Language Model**:
```python
from transformers import AutoModelForCausalLM

language_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
```

**Projection Layer**:
```python
import torch.nn as nn

projection = nn.Linear(
    vision_encoder.config.hidden_size,
    language_model.config.hidden_size
)
```

## Data Preparation for VLMs

### Image-Text Datasets

**Common Datasets**:
- **COCO**: Image captioning
- **Visual Genome**: Dense image descriptions
- **Flickr30k**: Image-text pairs
- **LAION**: Large-scale image-text pairs
- **VQA v2**: Visual question answering

**Load COCO Dataset**:
```python
from datasets import load_dataset

coco_dataset = load_dataset("HuggingFaceM4/COCO")
```

### Data Formatting

**Image-Text Pair Format**:
```python
{
    "image": <PIL.Image>,
    "text": "A cat sitting on a couch"
}
```

**VQA Format**:
```python
{
    "image": <PIL.Image>,
    "question": "What is in the image?",
    "answer": "A cat"
}
```

**Instruction Format**:
```python
{
    "instruction": "Describe this image in detail.",
    "image": <PIL.Image>,
    "output": "The image shows a cat sitting on a red couch..."
}
```

### Image Preprocessing

```python
from transformers import CLIPProcessor
from PIL import Image

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def preprocess_image_text(image, text):
    inputs = processor(
        text=text,
        images=image,
        return_tensors="pt",
        padding=True
    )
    return inputs
```

### Data Augmentation

**Image Augmentation**:
```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
])
```

**Text Augmentation**:
```python
# Paraphrasing for text
from transformers import pipeline

paraphraser = pipeline("text2text-generation", model="t5-small")

def augment_text(text):
    paraphrase = paraphraser(f"paraphrase: {text}")
    return paraphrase[0]['generated_text']
```

## VLM Fine-Tuning with LoRA

### Using LLaVA

**Load LLaVA Model**:
```python
from transformers import AutoProcessor, LlavaForConditionalGeneration

model = LlavaForConditionalGeneration.from_pretrained(
    "llava-hf/llava-1.5-7b-hf",
    torch_dtype=torch.float16,
    device_map="auto"
)

processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
```

**Apply LoRA**:
```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
```

### Training Data Preparation

```python
def prepare_vlm_dataset(examples):
    # Process images and text
    images = [img.convert("RGB") for img in examples["image"]]
    texts = examples["text"]
    
    # Create prompts
    prompts = [
        f"USER: <image>\nDescribe this image.\nASSISTANT: {text}"
        for text in texts
    ]
    
    # Process
    inputs = processor(
        text=prompts,
        images=images,
        return_tensors="pt",
        padding=True
    )
    
    # Set labels
    inputs["labels"] = inputs["input_ids"].clone()
    
    return inputs
```

### Training Loop

```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./vlm-finetune",
    num_train_epochs=3,
    per_device_train_batch_size=2,  # Smaller due to images
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    warmup_ratio=0.1,
    logging_steps=10,
    save_steps=500,
    fp16=True,
    gradient_checkpointing=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()
```

## Multimodal Task Types

### Image Captioning

**Task**: Generate descriptive text for images

**Example**:
```python
prompt = "USER: <image>\nDescribe this image in detail.\nASSISTANT:"

inputs = processor(text=prompt, images=image, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=200)
caption = processor.decode(outputs[0], skip_special_tokens=True)
```

### Visual Question Answering

**Task**: Answer questions about images

**Example**:
```python
question = "What is in this image?"
prompt = f"USER: <image>\n{question}\nASSISTANT:"

inputs = processor(text=prompt, images=image, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
answer = processor.decode(outputs[0], skip_special_tokens=True)
```

### Image-Text Retrieval

**Task**: Retrieve images given text or vice versa

**Using CLIP**:
```python
from transformers import CLIPModel, CLIPProcessor

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Encode text
text_inputs = processor(text=["a dog", "a cat"], return_tensors="pt")
text_features = model.get_text_features(**text_inputs)

# Encode image
image_inputs = processor(images=image, return_tensors="pt")
image_features = model.get_image_features(**image_inputs)

# Calculate similarity
similarity = (image_features @ text_features.T).softmax(dim=-1)
```

### Image Classification with Natural Language

**Task**: Classify images using natural language labels

```python
# Define labels
labels = ["dog", "cat", "bird", "car"]

# Encode labels
label_inputs = processor(text=labels, return_tensors="pt")
label_features = model.get_text_features(**label_inputs)

# Classify image
image_features = model.get_image_features(**image_inputs)
similarities = (image_features @ label_features.T).softmax(dim=-1)
predicted_class = labels[similarities.argmax()]
```

## Evaluation Metrics

### Captioning Metrics

**BLEU Score**:
```python
from datasets import load_metric
bleu = load_metric("bleu")

predictions = [["a", "dog", "sitting"]]
references = [["a", "dog", "is", "sitting"]]

results = bleu.compute(predictions=predictions, references=references)
```

**ROUGE Score**:
```python
rouge = load_metric("rouge")
results = rouge.compute(predictions=predictions, references=references)
```

**CIDEr Score** (for image captioning):
```python
# CIDEr is specific to image captioning
# Requires special implementation
```

### VQA Metrics

**Exact Match**:
```python
def exact_match(predicted, reference):
    return predicted.strip().lower() == reference.strip().lower()
```

**Accuracy**:
```python
def vqa_accuracy(predictions, references):
    correct = sum(
        exact_match(pred, ref)
        for pred, ref in zip(predictions, references)
    )
    return correct / len(predictions)
```

### Retrieval Metrics

**Recall@K**:
```python
def recall_at_k(retrieved_items, relevant_items, k):
    retrieved_k = retrieved_items[:k]
    relevant_retrieved = len(set(retrieved_k) & set(relevant_items))
    return relevant_retrieved / len(relevant_items)
```

**Mean Reciprocal Rank (MRR)**:
```python
def mean_reciprocal_rank(retrieved_items, relevant_items):
    for i, item in enumerate(retrieved_items):
        if item in relevant_items:
            return 1 / (i + 1)
    return 0
```

## Practical Applications

### Medical Image Analysis

**Use Case**: Analyze medical images with textual descriptions

```python
# Fine-tune on medical dataset
medical_dataset = load_dataset("path/to/medical_vlm")

# Train for medical image description
# Use specialized medical vocabulary
```

### E-commerce Product Description

**Use Case**: Generate product descriptions from images

```python
# Product image → Description
prompt = "USER: <image>\nDescribe this product for an e-commerce listing.\nASSISTANT:"

# Include product features
# Highlight selling points
# Use marketing language
```

### Accessibility

**Use Case**: Describe images for visually impaired users

```python
# Generate detailed, helpful descriptions
# Focus on important visual information
# Use clear, descriptive language
```

### Content Moderation

**Use Case**: Analyze images for policy violations

```python
# Detect inappropriate content
# Flag policy violations
# Provide reasoning
```

## Practical Exercise

### Exercise 1: Image Captioning

1. Load a small image-text dataset (100 examples)
2. Fine-tune LLaVA with LoRA
3. Generate captions for test images
4. Evaluate with BLEU and ROUGE

### Exercise 2: Visual Question Answering

1. Create VQA dataset (50 questions)
2. Fine-tune for VQA task
3. Answer questions about test images
4. Calculate accuracy

### Exercise 3: Custom Application

Choose an application (e.g., medical, e-commerce):
1. Collect or create dataset
2. Fine-tune VLM
3. Test on your use case
4. Evaluate performance

## Key Takeaways

1. **VLMs combine vision and language** understanding
2. **Multiple architectures** exist (CLIP, LLaVA, BLIP)
3. **Data preparation** requires image and text processing
4. **LoRA works for VLMs** just like language models
5. **Task-specific metrics** evaluate different VLM capabilities
6. **Practical applications** span many domains
7. **Fine-tuning adapts VLMs** to specific use cases

## Next Steps

Congratulations! You've completed all 9 chapters. You now have comprehensive knowledge of LLM fine-tuning from fundamentals to advanced topics like vision-language models.

## Additional Resources

- [LLaVA Paper](https://arxiv.org/abs/2310.03744)
- [CLIP Paper](https://arxiv.org/abs/2103.00020)
- [Hugging Face Multimodal Docs](https://huggingface.co/docs/transformers/multimodal)
- [Visual Genome Dataset](https://visualgenome.org/)