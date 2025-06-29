# Vocabulary Size Management in NMT Training

This document summarizes key considerations for managing the `vocab_size` parameter in Neural Machine Translation (NMT) models, particularly when using SentencePiece tokenization and scaling datasets.

## The `vocab_size` Challenge

When `vocab_size` is too small for a given dataset, common issues arise:

* Out-Of-Vocabulary (OOV) Tokens: A small vocabulary forces the tokenizer to represent many unique words/subwords as `<unk>` (unknown) tokens. This leads to significant information loss.
* Poor Training Signal: Models struggle to learn meaningful representations and translation patterns from data heavily populated by `<unk>` tokens.
* Tokenizer Quality: A constrained `vocab_size` results in a less effective SentencePiece model, leading to aggressive subword merging or overly generic tokens.
* Model Mismatch: The NMT model's embedding layer expects a vocabulary size matching the tokenizer's output. A mismatch or an overly small effective vocabulary limits the model's capacity to distinguish linguistic units.

## Updating `vocab_size` with Growing Data

`vocab_size` should directly reflect the diversity and scale of the linguistic data.

### 1. SentencePiece Training

* The configured `vocab_size` must be passed to the SentencePiece training process.
* The generated `.model` and `.vocab` files will contain exactly this number of tokens.
* The NMT model's `src_vocab_size` and `trg_vocab_size` (embedding dimensions and final output layer) must match the actual size of the SentencePiece vocabulary.

### 2. Scaling Guideline

* Initial / Tiny Data (e.g., <1MB): A `vocab_size` of 100-200 is problematic and likely leads to untrainable models due to excessive OOV. For initial, minimal data, a slightly higher value (e.g., 1,000-5,000) is a starting point, but genuine training requires more data.
* Small-to-Medium Data (MBs to tens of MBs): Target `vocab_size` between 8,000 and 16,000.
* Large Datasets (hundreds of MBs to GBs): Common `vocab_size` ranges from 32,000 to 64,000.

### 3. Process for Updating `vocab_size`

* Modify `config/data_config.yaml`: Adjust the `vocab_size` parameter to the desired new value.

```yaml
# Example data_config.yaml
vocab_size: 16000 # Example new value
```

* Retrain SentencePiece Model: Crucially, re-run the SentencePiece training script using the updated `vocab_size`. This generates new `.model` and `.vocab` files.
* Update Model Configuration: Ensure `src_vocab_size` and `trg_vocab_size` in `config/model_config.yaml` (or equivalent) are set to the new, actual vocabulary size.
* Re-process All Data: Re-tokenize all raw data splits (train, validation, test) using the newly trained SentencePiece model.
* Retrain NMT Model: Start NMT model training from scratch with the updated tokenization and model configuration.

By following this process, the model can leverage a richer vocabulary, leading to more stable training and improved translation quality as the dataset scales.
