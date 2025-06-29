#!/bin/bash

# Script to preprocess the data for the Zosia NMT project.
# This typically involves:
# 1. Training the SentencePiece tokenizer on raw data.
# 2. Tokenizing the raw text files into numerical IDs.
# 3. Saving the tokenized data in a format suitable for PyTorch DataLoaders.

echo "--- Starting Data Preprocessing for Zosia ---"

# Ensure the 'data' directory and its subdirectories exist if your make_dataset.py doesn't create them.
# Usually, make_dataset.py handles this, but it's good to be aware.
# mkdir -p data/raw data/processed data/vocab

# Execute the make_dataset.py module to run the preprocessing pipeline.
# It should read configurations from config/data_config.yaml.
python -m src.data.make_dataset --data_config config/data_config.yaml

if [ $? -eq 0 ]; then
    echo "--- Data Preprocessing Completed Successfully ---"
else
    echo "--- Data Preprocessing Failed ---"
    exit 1
fi