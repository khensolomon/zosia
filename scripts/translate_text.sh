#!/bin/bash
# -----------------------------------------------------------------------------
# Script: translate_text.sh
#
# Description:
#   This script uses a trained model checkpoint to translate a given text.
#   It now supports beam search with a length penalty alpha.
#
# Usage:
#   bash scripts/translate_text.sh \
#     --model_file <path> \
#     --text "Your text" \
#     --src <lang> \
#     --tgt <lang> \
#     [--beam_size 5] \
#     [--alpha 0.6]
#
# Arguments:
#   --model_file: Path to the trained model checkpoint (.pt file).
#   --text: The sentence to translate.
#   --src: The source language code.
#   --tgt: The target language code.
#   --beam_size (optional): The number of beams for beam search.
#                         Defaults to 1 (greedy decoding).
#   --alpha (optional): The length penalty strength. 0=no penalty.
#                     A common value is 0.6.
# -----------------------------------------------------------------------------

set -e

# Initialize variables
MODEL_FILE=""
TEXT_TO_TRANSLATE=""
SRC_LANG=""
TGT_LANG=""
BEAM_SIZE="1"
ALPHA="0.6" # Default alpha for length penalty

# --- Argument Parsing ---
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model_file) MODEL_FILE="$2"; shift ;;
        --text) TEXT_TO_TRANSLATE="$2"; shift ;;
        --src) SRC_LANG="$2"; shift ;;
        --tgt) TGT_LANG="$2"; shift ;;
        --beam_size) BEAM_SIZE="$2"; shift ;;
        --alpha) ALPHA="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# --- Validation ---
if [ -z "$MODEL_FILE" ] || [ -z "$TEXT_TO_TRANSLATE" ] || [ -z "$SRC_LANG" ] || [ -z "$TGT_LANG" ]; then
    echo "Error: Missing required arguments."
    exit 1
fi

# --- Script Body ---
echo "============================================="
echo "        ZoSia Interactive Translator         "
echo "============================================="
echo
echo "Model:         ${MODEL_FILE}"
echo "Direction:     ${SRC_LANG} -> ${TGT_LANG}"
if [ "$BEAM_SIZE" -gt 1 ]; then
    echo "Search Method: Beam Search (size=${BEAM_SIZE}, alpha=${ALPHA})"
else
    echo "Search Method: Greedy Decoding"
fi
echo "Input Text:    \"${TEXT_TO_TRANSLATE}\""
echo "---------------------------------------------"
echo

# Launch the Python translation module
python -m src.translate.translator \
    --model_file "$MODEL_FILE" \
    --text "$TEXT_TO_TRANSLATE" \
    --src_lang "$SRC_LANG" \
    --tgt_lang "$TGT_LANG" \
    --beam_size "$BEAM_SIZE" \
    --alpha "$ALPHA"

echo
echo "============================================="
echo "           Translation Complete            "
echo "============================================="
