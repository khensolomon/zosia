#!/bin/bash

# Script to translate a given text using the Zosia NMT model.
# It uses the 'latest_run' symbolic link to automatically find the most
# recently trained best model.

# Usage: ./scripts/translate_text.sh "Your sentence to translate."

# Check if a sentence was provided as an argument
if [ -z "$1" ]; then
    echo "Usage: ./scripts/translate_text.sh \"Your sentence to translate.\""
    echo "Example: ./scripts/translate_text.sh \"Dam maw?\""
    exit 1
fi

SENTENCE_TO_TRANSLATE="$1"
# Path to the best model checkpoint, using the 'latest_run' symlink.
# This assumes your training script correctly sets up 'experiments/latest_run'
# pointing to the most recent run, and that 'best_model.pt' is within its 'checkpoints' folder.
MODEL_CHECKPOINT="experiments/latest_run/checkpoints/best_model.pt"

echo "--- Starting Zosia Text Translation ---"
echo "Input Sentence: \"$SENTENCE_TO_TRANSLATE\""

# Execute the translator.py module.
# It should read configurations and load the model from the specified checkpoint path.
python -m src.translate.translator \
    --checkpoint_path "$MODEL_CHECKPOINT" \
    --sentence "$SENTENCE_TO_TRANSLATE"

if [ $? -eq 0 ]; then
    echo "--- Translation Completed ---"
else
    echo "--- Translation Failed ---"
    exit 1
fi