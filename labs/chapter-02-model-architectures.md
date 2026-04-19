# Chapter 2: Model Architectures

## Learning Objectives

By the end of this chapter, you will be able to:
- Understand the transformer architecture fundamentals
- Compare different LLM families and their characteristics
- Evaluate model size vs. performance trade-offs
- Understand model parameters and their roles
- Select appropriate architectures for specific tasks

## Duration

60-90 minutes

## Prerequisites

- Chapter 1: From Generalist to Specialist
- Basic understanding of neural networks
- Familiarity with Python

## Introduction

Understanding model architectures is crucial for effective fine-tuning. This chapter covers the fundamentals of transformer architecture, compares popular LLM families, and helps you choose the right architecture for your needs.

## Transformer Architecture Fundamentals

### Core Components

**1. Self-Attention Mechanism**

The self-attention mechanism allows the model to weigh the importance of different words in a sequence when processing each word.

```python
# Simplified attention concept
attention_scores = Query × Key^T
attention_weights = softmax(attention_scores)
output = attention_weights × Value
```

**Key Concepts**:
- **Query (Q)**: What the current position is looking for
- **Key (K)**: What each position offers
- **Value (V)**: The actual information at each position
- **Multi-head attention**: Multiple attention heads learn different patterns

**2. Position Encoding**

Since transformers process sequences in parallel, they need positional information:
- **Absolute positional encoding**: Fixed or learned position embeddings
- **Relative positional encoding**: Encodes relative positions
- **RoPE (Rotary Positional Embeddings)**: Modern approach used in LLaMA 2/3

**3. Feed-Forward Networks**

Each transformer layer includes a feed-forward network:
- Two linear transformations with a non-linearity
- Expands then contracts the dimensionality
- Captures complex patterns beyond attention

**4. Layer Normalization**

Normalizes activations to stabilize training:
- Pre-norm vs. post-norm architectures
- Helps with gradient flow
- Enables deeper networks

### Architecture Variants

**Encoder-Decoder (Original Transformer)**:
- Encoder processes input sequence
- Decoder generates output sequence
- Used in machine translation (T5, BART)

**Decoder-Only (GPT-style)**:
- Single stack of decoder layers
- Autoregressive generation
- Used in most LLMs (GPT, LLaMA, Mistral)

**Encoder-Only (BERT-style)**:
- Single stack of encoder layers
- Bidirectional context
- Used for understanding tasks (BERT, RoBERTa)

## LLM Families

### GPT Family (OpenAI)

**GPT-3/3.5/4**:
- Decoder-only architecture
- Trained on massive web text
- Strong general-purpose performance
- Proprietary (API access only)

**Characteristics**:
- Excellent generation capabilities
- Good at following instructions
- Strong reasoning abilities
- Limited fine-tuning access

### LLaMA Family (Meta)

**LLaMA 2**:
- Open weights (research license)
- 7B, 13B, 70B parameter variants
- Strong performance for size
- Good for fine-tuning

**LLaMA 3**:
- Improved performance
- 8B, 70B variants
- More permissive license
- Better efficiency

**CodeLLaMA**:
- Specialized for code
- Python, Python, and CodeLlama 34B
- Excellent for programming tasks

**Characteristics**:
- Strong open-source option
- Good documentation
- Active community
- Commercial use restrictions (check license)

### Mistral Family

**Mistral 7B**:
- Efficient architecture
- Sliding window attention
- Grouped query attention
- Outperforms larger models

**Mixtral 8x7B**:
- Mixture of Experts (MoE)
- 8 experts, 2 active per token
- Efficient inference
- Strong performance

**Characteristics**:
- Very efficient
- Apache 2.0 license
- Good documentation
- Active development

### Other Notable Families

**BLOOM** (BigScience):
- Multilingual focus
- Open license
- 176B parameters

**Falcon**:
- Efficient training
- Strong performance
- Apache 2.0 license

**Phi-2/Phi-3** (Microsoft):
- Small but capable
- Good for edge deployment
- Efficient architecture

## Model Size vs. Performance

### Parameter Count Impact

**General Trends**:
- More parameters → better performance (diminishing returns)
- Larger models need more memory and compute
- Smaller models can be specialized via fine-tuning

**Performance Curves**:
```
Performance ∝ log(Parameters)
```

**Trade-offs**:
- 7B: Good balance, runs on consumer GPUs
- 13B: Better performance, needs more memory
- 34B: Strong performance, needs professional GPUs
- 70B: State-of-the-art, needs significant resources

### Efficient Architectures

**Techniques for Efficiency**:

1. **Mixture of Experts (MoE)**
   - Mixtral uses 8 experts
   - Only 2 experts active per token
   - Efficient inference

2. **Sliding Window Attention**
   - Limited context window
   - Reduces computational cost
   - Mistral uses this

3. **Grouped Query Attention**
   - Reduces attention parameters
   - Maintains performance
   - Used in Mistral and LLaMA 3

4. **Flash Attention**
   - Optimized attention implementation
   - Reduces memory usage
   - Faster training/inference

## Understanding Model Parameters

### Parameter Types

**1. Embedding Parameters**
- Token embeddings: Map tokens to vectors
- Position embeddings: Encode position information
- Typically 10-20% of total parameters

**2. Attention Parameters**
- Query, Key, Value projections
- Output projection
- Attention bias terms
- Typically 30-40% of total parameters

**3. Feed-Forward Parameters**
- Two linear transformations
- Expansion and contraction
- Typically 40-50% of total parameters

**4. Layer Norm Parameters**
- Scale and shift parameters
- Small percentage (<5%)

### Parameter Count Calculation

**For a simple transformer layer**:
```
Params per layer = (d_model × d_model × 4) + (d_model × 4 × d_ff) + layer_norm
```

**Where**:
- d_model: Model dimension (e.g., 4096)
- d_ff: Feed-forward dimension (e.g., 4 × d_model)

**Example: 7B Model**
- 32 layers
- d_model = 4096
- d_ff = 16384
- ~7B total parameters

## Architecture Selection for Tasks

### Text Generation

**Recommended Architectures**:
- LLaMA 2/3 (strong generation)
- Mistral (efficient)
- GPT-style models

**Considerations**:
- Generation quality
- Coherence and fluency
- Instruction following

### Code Generation

**Recommended Architectures**:
- CodeLLaMA
- StarCoder
- Code-specific models

**Considerations**:
- Code syntax understanding
- Multi-language support
- Code completion accuracy

### Question Answering

**Recommended Architectures**:
- LLaMA 2/3
- Mistral
- Encoder-decoder models (T5)

**Considerations**:
- Factual accuracy
- Context understanding
- Answer quality

### Multilingual Tasks

**Recommended Architectures**:
- BLOOM (multilingual focus)
- LLaMA 3 (improved multilingual)
- mGPT

**Considerations**:
- Language coverage
- Cross-lingual transfer
- Performance balance

## Practical Exercise

### Exercise 1: Architecture Comparison

Compare LLaMA 2 7B vs. Mistral 7B:

**LLaMA 2 7B**:
- 32 layers, 4096 dimension
- Standard attention
- Trained on 2T tokens

**Mistral 7B**:
- 32 layers, 4096 dimension
- Sliding window attention, GQA
- Trained on unknown tokens

**Questions**:
1. What architectural differences exist?
2. How might these affect performance?
3. Which would you choose for text generation?
4. Which would you choose for low-latency applications?

### Exercise 2: Parameter Analysis

Given a model with:
- 24 layers
- d_model = 3072
- d_ff = 12288
- 32 attention heads

**Calculate**:
1. Approximate parameter count
2. Memory requirements for FP16
3. Memory requirements for QLoRA (4-bit)

## Key Takeaways

1. **Transformer architecture** is the foundation of modern LLMs
2. **Self-attention** is the key mechanism for understanding context
3. **Different LLM families** have different strengths and licenses
4. **Model size matters** but diminishing returns exist
5. **Efficient architectures** (MoE, sliding window) improve performance
6. **Parameter distribution** affects memory and compute requirements
7. **Task-specific selection** is crucial for optimal results

## Next Steps

Now that you understand model architectures, let's set up your fine-tuning environment in [Chapter 3: The Fine-Tuner's Workshop](chapter-03-fine-tuners-workshop.md).

## Additional Resources

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer paper
- [LLaMA 2 Paper](https://arxiv.org/abs/2307.09288)
- [Mistral 7B Paper](https://arxiv.org/abs/2310.06825)
- [Hugging Face Transformers Docs](https://huggingface.co/docs/transformers)