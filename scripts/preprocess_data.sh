#!/bin/bash
# -----------------------------------------------------------------------------
# Script: preprocess_data.sh
#
# Description:
#   This script orchestrates the data preprocessing pipeline. It now accepts
#   language direction arguments to align with the training script.
#
# Usage:
#   Run this script from the root directory of the project.
#
#   To preprocess for en -> zo (default):
#   bash scripts/preprocess_data.sh
#
#   To preprocess for zo -> en:
#   bash scripts/preprocess_data.sh --src zo --tgt en
# -----------------------------------------------------------------------------

# --- Configuration ---
set -e
SRC_LANG="en"
TGT_LANG="zo"

# --- Argument Parsing ---
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
echo "  Starting ZoSia Data Preprocessing Pipeline "
echo "        Direction: ${SRC_LANG} -> ${TGT_LANG}"
echo "============================================="
echo

# Step 1: Train Tokenizers (This is direction-agnostic and only needs to be run once)
echo "[Step 1/2] Training tokenizers..."
python -m src.tokenizers.builder
echo "✅ Tokenizers trained successfully."
echo

# Step 2: Process and Split Datasets for the specified direction
echo "[Step 2/2] Processing and splitting datasets..."
python -m src.dataset.builder --src_lang "${SRC_LANG}" --tgt_lang "${TGT_LANG}"
echo "✅ Datasets processed and split successfully."
echo

echo "============================================="
echo "     Preprocessing Pipeline Complete!        "
echo "============================================="
