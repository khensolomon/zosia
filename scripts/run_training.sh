#!/bin/bash

# Script to run the training process for the Zosia NMT model.
# This script starts the main training loop as defined in src/train/trainer.py.

echo "--- Starting Zosia Model Training ---"

# Ensure the 'experiments' directory exists
mkdir -p experiments

# Execute the trainer.py module.
# It will read the necessary configurations from the specified YAML files.
# The trainer is also responsible for creating the timestamped experiment directory
# and the 'latest_run' symbolic link within the 'experiments' directory.
python -m src.train.trainer \
    --config config/training_config.yaml \
    --model_config config/model_config.yaml \
    --data_config config/data_config.yaml

if [ $? -eq 0 ]; then
    echo "--- Zosia Model Training Completed Successfully ---"
    echo "Check 'experiments/latest_run' for the latest run's outputs."
else
    echo "--- Zosia Model Training Failed ---"
    exit 1
fi