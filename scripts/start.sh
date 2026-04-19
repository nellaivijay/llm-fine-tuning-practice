#!/bin/bash

# Start Script for LLM Fine-Tuning Practice
# This script starts Jupyter and monitoring services

set -e

echo "Starting LLM Fine-Tuning Practice services..."

# Activate virtual environment
if [ -d "llm-env" ]; then
    source llm-env/bin/activate
else
    echo "Error: Virtual environment not found. Run setup.sh first."
    exit 1
fi

# Start Jupyter Notebook
echo "Starting Jupyter Notebook..."
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password='' &
JUPYTER_PID=$!

echo "Jupyter Notebook started with PID: $JUPYTER_PID"
echo "Access at: http://localhost:8888"

# Save PID for later
echo $JUPYTER_PID > logs/jupyter.pid

echo "Services started successfully!"
echo ""
echo "To stop services, run: ./scripts/stop.sh"
echo ""
echo "To check service health, run: ./scripts/health-check.sh"