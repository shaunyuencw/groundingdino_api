#!/bin/bash

# Activate the virtual environment
CONDA_PATH=/home/shaun/miniconda3
ENV_NAME=gdino_api

# Initialize Conda
source $CONDA_PATH/etc/profile.d/conda.sh

# Activate the environment
conda activate $ENV_NAME

# Ensure the logs directory exists
mkdir -p logs

# Start Gunicorn servers for each GPU
for i in {0..3}
do
    CUDA_VISIBLE_DEVICES=$i GPU_ID=$i gunicorn -w 1 -t 120 -k uvicorn.workers.UvicornWorker app:app --bind 0.0.0.0:$(expr 8001 + $i) &> logs/gunicorn_$(expr 8001 + $i).log &
done

# Note: Removed the 'wait' command to avoid waiting for background processes
echo "Servers are starting... check logs for details."
