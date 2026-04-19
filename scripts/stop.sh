#!/bin/bash

# Stop Script for LLM Fine-Tuning Practice
# This script stops running services

set -e

echo "Stopping LLM Fine-Tuning Practice services..."

# Stop Jupyter if running
if [ -f "logs/jupyter.pid" ]; then
    JUPYTER_PID=$(cat logs/jupyter.pid)
    if ps -p $JUPYTER_PID > /dev/null; then
        echo "Stopping Jupyter Notebook (PID: $JUPYTER_PID)..."
        kill $JUPYTER_PID
        rm logs/jupyter.pid
        echo "Jupyter Notebook stopped"
    else
        echo "Jupyter Notebook not running"
        rm logs/jupyter.pid
    fi
else
    echo "No Jupyter PID file found"
fi

# Stop any other services if needed
# Add additional service stop commands here

echo "All services stopped successfully!"