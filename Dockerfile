FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /workspace/
RUN pip install --no-cache-dir -r requirements.txt

# Install Jupyter
RUN pip install jupyter jupyterlab

# Create directories
RUN mkdir -p /workspace/notebooks \
    /workspace/data \
    /workspace/models \
    /workspace/outputs \
    /workspace/checkpoints \
    /workspace/logs \
    /workspace/config

# Copy configuration files
COPY .env.example /workspace/.env.example

# Set environment variables
ENV PYTHONPATH=/workspace
ENV JUPYTER_ENABLE_LAB=yes

# Expose Jupyter port
EXPOSE 8888

# Default command
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]