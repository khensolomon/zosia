#!/bin/bash
# -----------------------------------------------------------------------------
# Script: run_training.sh
#
# Description:
#   This script launches the model training process for a specific language
#   pair. It allows specifying the source and target languages, enabling
#   bidirectional training.
#
# Usage:
#   Run this script from the root directory of the project.
#
#   To train with default direction (en -> zo):
#   bash scripts/run_training.sh
#
#   To train with a specific direction (e.g., zo -> en):
#   bash scripts/run_training.sh --src zo --tgt en
#
# Requirements:
#   - A configured environment with all dependencies from requirements.txt.
#   - Preprocessed data must exist in `data/processed/`. This is generated
#     by running `scripts/preprocess_data.sh`.
# -----------------------------------------------------------------------------

# --- Configuration ---
# Exit immediately if a command exits with a non-zero status.
set -e

# Default values for source and target languages
SRC_LANG="en"
TGT_LANG="zo"

# --- Argument Parsing ---
# This loop parses command-line arguments to override the default languages.
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --src) SRC_LANG="$2"; shift ;;
        --tgt) TGT_LANG="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# --- Script Body ---
echo "============================================="
echo "      Starting ZoSia Model Training          "
echo "============================================="
echo
echo "Translation Direction: ${SRC_LANG} -> ${TGT_LANG}"
echo "Start Time: $(date)"
echo

# Launch the Python training module, passing the language pair as arguments.
# The Python script will handle loading the correct configurations and data.
python -m src.train.trainer --src_lang "${SRC_LANG}" --tgt_lang "${TGT_LANG}"

# Check the exit code of the Python script
if [ $? -eq 0 ]; then
    echo
    echo "============================================="
    echo "      Training Completed Successfully!       "
    echo "============================================="
    echo "End Time: $(date)"
    echo "Find model checkpoints and logs in the 'experiments' directory."
else
    echo
    echo "============================================="
    echo "      Training Failed. Please check logs.    "
    echo "============================================="
    echo "End Time: $(date)"
    exit 1
fi
