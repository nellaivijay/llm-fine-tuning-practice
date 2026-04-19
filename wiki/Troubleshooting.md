# Troubleshooting

Common issues and solutions for LLM Fine-Tuning Practice.

## GPU Issues

### CUDA Not Available

**Symptoms**: `CUDA not available` error when importing PyTorch

**Solution**:
```bash
# Check CUDA installation
nvcc --version

# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify
python -c "import torch; print(torch.cuda.is_available())"
```

### GPU Not Detected

**Symptoms**: `CUDA available: False` despite having GPU

**Solution**:
```bash
# Check NVIDIA driver
nvidia-smi

# Reinstall NVIDIA drivers
sudo apt-get install nvidia-driver-535

# Restart system
sudo reboot
```

### Out of Memory

**Symptoms**: `CUDA out of memory` error during training

**Solutions**:
1. **Use QLoRA**:
   ```python
   from transformers import BitsAndBytesConfig
   bnb_config = BitsAndBytesConfig(
       load_in_4bit=True,
       bnb_4bit_quant_type="nf4"
   )
   ```

2. **Reduce batch size**:
   ```python
   training_args = TrainingArguments(
       per_device_train_batch_size=2,  # Reduce from 4
   )
   ```

3. **Use gradient accumulation**:
   ```python
   training_args = TrainingArguments(
       per_device_train_batch_size=2,
       gradient_accumulation_steps=8,  # Increase to maintain effective batch size
   )
   ```

4. **Enable gradient checkpointing**:
   ```python
   training_args = TrainingArguments(
       gradient_checkpointing=True,
   )
   ```

## Installation Issues

### Dependency Conflicts

**Symptoms**: `pip install` fails with dependency conflicts

**Solution**:
```bash
# Create fresh virtual environment
python3 -m venv fresh-env
source fresh-env/bin/activate

# Install PyTorch first
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Then install other dependencies
pip install -r requirements.txt
```

### Permission Errors

**Symptoms**: `Permission denied` when writing files

**Solution**:
```bash
# Fix directory permissions
chmod -R 755 data models outputs checkpoints logs

# Or run with sudo (not recommended)
sudo python your_script.py
```

### Hugging Face Authentication

**Symptoms**: `OSError: mistralai/Mistral-7B-v0.1 is a gated model`

**Solution**:
```bash
# Login to Hugging Face
huggingface-cli login

# Enter your access token
# Get token from: https://huggingface.co/settings/tokens
```

## Training Issues

### Training Slow

**Symptoms**: Training takes much longer than expected

**Solutions**:
1. **Enable mixed precision**:
   ```python
   training_args = TrainingArguments(fp16=True)
   ```

2. **Use smaller model**:
   ```python
   # Use 7B instead of 13B or 70B
   ```

3. **Optimize data loading**:
   ```python
   from torch.utils.data import DataLoader
   dataloader = DataLoader(dataset, batch_size=32, num_workers=4)
   ```

4. **Check GPU utilization**:
   ```bash
   watch -n 1 nvidia-smi
   ```

### Loss Not Decreasing

**Symptoms**: Training loss stays high or increases

**Solutions**:
1. **Check learning rate**:
   ```python
   training_args = TrainingArguments(
       learning_rate=2e-4,  # Try different values
   )
   ```

2. **Verify data quality**:
   - Check for data errors
   - Ensure proper formatting
   - Remove bad examples

3. **Check model initialization**:
   ```python
   # Ensure LoRA is properly applied
   model.print_trainable_parameters()
   ```

4. **Increase training epochs**:
   ```python
   training_args = TrainingArguments(
       num_train_epochs=5,  # Increase from 3
   )
   ```

### Model Not Saving

**Symptoms**: Checkpoints not being saved

**Solutions**:
1. **Check save strategy**:
   ```python
   training_args = TrainingArguments(
       save_strategy="steps",
       save_steps=500,
   )
   ```

2. **Check disk space**:
   ```bash
   df -h
   ```

3. **Check permissions**:
   ```bash
   chmod -R 755 outputs checkpoints
   ```

## Data Issues

### Data Loading Errors

**Symptoms**: Errors when loading or processing data

**Solutions**:
1. **Check data format**:
   ```python
   # Verify data structure
   print(dataset[0])
   ```

2. **Handle missing values**:
   ```python
   df = df.dropna()
   ```

3. **Check file paths**:
   ```python
   import os
   print(os.path.exists("data/train.jsonl"))
   ```

### Tokenization Errors

**Symptoms**: Errors during tokenization

**Solutions**:
1. **Set padding token**:
   ```python
   tokenizer.pad_token = tokenizer.eos_token
   ```

2. **Check sequence length**:
   ```python
   training_args = TrainingArguments(
       max_length=512,  # Reduce if too long
   )
   ```

3. **Handle special tokens**:
   ```python
   tokenizer.add_special_tokens({'pad_token': '[PAD]'})
   ```

## Evaluation Issues

### Metrics Not Calculating

**Symptoms**: Evaluation metrics return None or error

**Solutions**:
1. **Check compute_metrics function**:
   ```python
   def compute_metrics(eval_pred):
       predictions, labels = eval_pred
       # Ensure proper format
       return {"accuracy": accuracy_score(labels, predictions)}
   ```

2. **Verify evaluation dataset**:
   ```python
   print(len(eval_dataset))
   print(eval_dataset[0])
   ```

### Poor Evaluation Results

**Symptoms**: Model performs poorly on evaluation

**Solutions**:
1. **Check for overfitting**:
   - Compare train vs. eval loss
   - Add regularization if needed

2. **Increase training data**:
   - Collect more examples
   - Use synthetic data

3. **Adjust hyperparameters**:
   - Learning rate
   - Batch size
   - Model size

## Jupyter Notebook Issues

### Kernel Crashing

**Symptoms**: Jupyter kernel crashes when running cells

**Solutions**:
1. **Reduce memory usage**:
   - Use smaller batches
   - Clear variables: `del variable`
   - Restart kernel periodically

2. **Increase memory limit**:
   ```bash
   # In Jupyter notebook
   !python -c "import torch; print(torch.cuda.get_device_properties(0).total_memory)"
   ```

3. **Use Colab**:
   - Free GPU access
   - Higher memory limits

### Import Errors

**Symptoms**: `ModuleNotFoundError` in notebooks

**Solutions**:
1. **Install in correct environment**:
   ```bash
   # Activate virtual environment
   source llm-env/bin/activate
   
   # Install jupyter in environment
   pip install jupyter jupyterlab
   ```

2. **Check kernel**:
   - Kernel → Change kernel → Select correct environment

## Cloud Issues

### AWS Instance Issues

**Symptoms**: Problems with AWS EC2 instances

**Solutions**:
1. **Check instance type**:
   - Use p3/p4 instances for GPU
   - Ensure GPU availability in region

2. **Check security groups**:
   - Allow SSH (port 22)
   - Allow Jupyter (port 8888)

3. **Monitor costs**:
   - Use spot instances for savings
   - Stop instances when not in use

### Colab Issues

**Symptoms**: Problems with Google Colab

**Solutions**:
1. **Enable GPU**:
   - Runtime → Change runtime type → GPU

2. **Handle timeouts**:
   - Save checkpoints frequently
   - Use smaller models
   - Consider Colab Pro

3. **Mount Drive**:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

## Getting Help

If you're still stuck:

1. Check the [Installation Guide](Installation-Guide.md)
2. Review the lab documentation
3. Search existing GitHub issues
4. Open a new issue with details:
   - Error message
   - System information
   - Steps to reproduce

## Prevention Tips

### Regular Maintenance

```bash
# Clean up old checkpoints
rm -rf checkpoints/old-*/

# Clean up logs
rm -rf logs/*.log

# Clean up Docker
docker system prune -f
```

### Monitoring

```bash
# Regular GPU checks
watch -n 1 nvidia-smi

# Disk usage
df -h

# Memory usage
free -h
```

### Backup

```bash
# Backup important data
cp -r data/ data_backup/
cp -r models/ models_backup/
cp -r checkpoints/ checkpoints_backup/
```

---

**Still having issues? Open a GitHub issue with details!**