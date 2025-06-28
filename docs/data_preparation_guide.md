# Data Preparation Guide

This guide details the process of preparing data for the Neural Machine Translation (NMT) project, from raw text acquisition to tokenization and final processing for model training.

## 1. Data Structure Overview

The project expects a specific data directory structure:

```bash
data/
├── raw/
│   ├── base_training.yaml        # Catalog of raw data files
│   ├── parallel_base/
│   │   ├── train.en
│   │   ├── train.zo
│   │   ├── val.en
│   │   ├── val.zo
│   │   └── test.en
│   │   └── test.zo
│   └── ...                       # Other raw data directories (e.g., from different sources)
├── monolingual/
│   ├── zo/
│   │   ├── index.csv             # Index of monolingual Zomi files
│   │   ├── news.txt
│   │   └── bible.txt
│   └── en/
│       └── ...                   # Monolingual English data (if any)
├── processed/                    # Output directory for tokenized data (.pt files)
└── vocab/                        # Output directory for SentencePiece model (.model, .vocab)
```

## 2. Raw Data Acquisition

Acquire parallel text corpora (sentences aligned between Zomi and English) and any relevant monolingual data.

* **Parallel Data:** Store source and target language pairs in separate `.txt` files (e.g., `train.zo`, `train.en`). These files must have the same number of lines, with each line in one file corresponding to its translation in the other.
* **Monolingual Data:** Store monolingual text data in `.txt` files. This data can be used to enrich the SentencePiece tokenizer's vocabulary.

## 3. Configuring Raw Data Sources (`base_training.yaml`)

The `data/raw/base_training.yaml` file acts as a catalog, explicitly defining which raw data files are used for different splits (training, validation, testing) of the parallel corpus.

**Example `data/raw/base_training.yaml` content:**

```yaml
# Define parallel data sources for different splits
parallel_data:
  train:
    - description: "Base training set"
      path_en: "parallel_base/train.en"
      path_zo: "parallel_base/train.zo"
    # - description: "Additional training data" # Example for multiple sources
    #   path_en: "another_corpus/train_part2.en"
    #   path_zo: "another_corpus/train_part2.zo"
  val:
    - description: "Base validation set"
      path_en: "parallel_base/val.en"
      path_zo: "parallel_base/val.zo"
  test:
    - description: "Base test set"
      path_en: "parallel_base/test.en"
      path_zo: "parallel_base/test.zo"
```

* parallel_data: A top-level key containing definitions for parallel data.
* train, val, test: Keys representing different data splits. Each can be a list of dictionaries, allowing multiple sources per split.
* description: An optional human-readable description for each data source.
* path_en, path_zo: Paths to the English and Zomi files, respectively, relative to the data/raw/ directory.

## 4. Configuring Monolingual Data Sources (index.csv)

Monolingual data, especially for the target language or for both languages if available, can significantly improve tokenizer quality. The data/monolingual/zo/index.csv file should list Zomi monolingual files.

Example data/monolingual/zo/index.csv content:

```csv
filename,description
news.txt,News articles in Zomi
bible.txt,Zomi Bible translation
context.txt,Contextual Zomi sentences
```

* filename: Path to the monolingual file, relative to the data/monolingual/zo/ directory.
* description: An optional description.

## 5. Main Data Configuration (config/data_config.yaml)

This file holds global data-related parameters.

Key parameters to verify/adjust:

```yaml
raw_data_dir: "data/raw"             # Base directory for raw data
raw_data_catalog_file: "base_training.yaml" # Catalog file defining parallel data sources

monolingual_data_dir: "data/monolingual" # Base dir for monolingual data
monolingual_zo_index_file: "zo/index.csv" # Index for Zomi monolingual data

processed_data_dir: "data/processed" # Output directory for tokenized data
vocab_dir: "data/vocab"              # Output directory for tokenizer model
tokenizer_prefix: "zosia_sp"   # Prefix for SentencePiece model files

max_sequence_length: 128             # Maximum length for tokenized sequences
vocab_size: 16000                    # Desired size of the SentencePiece vocabulary
# ... other cleaning/filtering parameters
```

## 6. Running the Data Preparation Script

The make_dataset.py script orchestrates the entire data preparation pipeline:

Reads data_config.yaml to find input/output directories and parameters.

Loads base_training.yaml to identify all parallel data files.

Loads index.csv (for monolingual data) to identify additional text for tokenizer training.

Combines all specified parallel and monolingual text into a single corpus for SentencePiece training.

Trains a SentencePiece BPE (Byte Pair Encoding) tokenizer based on the vocab_size and tokenizer_prefix specified in data_config.yaml.

Tokenizes all parallel data splits (train, val, test) using the trained SentencePiece model.

Saves the tokenized data for each split as PyTorch .pt files (e.g., train_token_ids.zo.pt, train_token_ids.en.pt) in the data/processed/ directory.

To run the data preparation:

```bash
python -m src.data.make_dataset
```

## 7. Data Loading for Training (src/data/dataset_utils.py)

The src/data/dataset_utils.py module contains the NMTDataset class and get_dataloaders function responsible for loading the processed .pt files during training.

* This module reads the raw_data_catalog_file from data_config.yaml to determine the expected filenames and locations of the processed .pt files.
* It then loads these files, creates NMTDataset instances, and wraps them in DataLoaders for efficient batching during model training.

Crucial Note on Data Volume: Neural Machine Translation models, especially Transformer-based architectures, are highly data-intensive. A sufficient quantity of high-quality parallel data (typically tens to hundreds of thousands of sentence pairs or more) is essential for the model to learn meaningful translation patterns and achieve reasonable performance. Running training with very small datasets (e.g., tens of samples) will result in trivial or zero BLEU scores and high perplexity, as the model lacks enough examples to generalize.

### 4. `docs/model_architecture.md`

```markdown
# Model Architecture: Transformer for NMT

This document provides an overview of the Transformer model architecture implemented in this Neural Machine Translation (NMT) project, explaining its key components and how they contribute to the translation task.

## 1. Introduction to the Transformer

The Transformer is a deep learning model introduced in the paper "Attention Is All You Need" (Vaswani et al., 2017). It revolutionized sequence-to-sequence tasks like NMT by entirely eschewing recurrent and convolutional layers in favor of self-attention mechanisms. This design allows for parallel processing of input sequences, leading to significant speedups in training and improved performance compared to earlier RNN-based models.

## 2. Overall Structure

The Transformer model for NMT follows an Encoder-Decoder architecture:

* **Encoder:** Processes the source language input sequence to create a rich, context-aware representation.
* **Decoder:** Takes the Encoder's output representation and generates the target language output sequence, one token at a time.

Both the Encoder and Decoder are composed of multiple identical layers stacked on top of each other.

## 3. Key Components

### 3.1. Token Embedding

* **Purpose:** Converts input token IDs (from the SentencePiece tokenizer) into continuous vector representations. Each unique token in the vocabulary is mapped to a dense vector of a specified `d_model` dimension.
* **Location in Code:** Typically implemented as `torch.nn.Embedding` within a `TokenEmbedding` module.

### 3.2. Positional Encoding

* **Purpose:** Since the Transformer contains no recurrence or convolution, it needs a mechanism to inject information about the relative or absolute position of tokens in the sequence. Positional encodings are added to the token embeddings.
* **Mechanism:** Uses sine and cosine functions of different frequencies.
* **Location in Code:** Often a separate `PositionalEncoding` module that applies dropout.

### 3.3. Multi-Head Attention

This is the core building block of the Transformer. It allows the model to weigh the importance of different parts of the input sequence (or previous parts of the output sequence) when processing a token.

* **Scaled Dot-Product Attention:** The fundamental attention mechanism. It calculates attention scores by taking the dot product of a Query vector with Key vectors, scaling by the square root of the dimension of keys, and applying a softmax function. These scores are then used to weight a sum of Value vectors.
* **Multi-Head:** Instead of performing a single attention function, the Query, Key, and Value vectors are linearly projected `n_heads` times. Each "head" then performs attention in parallel. The results from these multiple heads are concatenated and linearly transformed, allowing the model to jointly attend to information from different representation subspaces at different positions.
* **Types of Attention in Transformer:**
    * **Encoder Self-Attention:** In the Encoder layers, each token attends to all other tokens in the *same* source sequence.
    * **Decoder Self-Attention (Masked):** In the Decoder layers, each token attends to all previous tokens in the *same* target sequence. It's "masked" to prevent attending to future tokens, ensuring causality.
    * **Encoder-Decoder Attention:** In the Decoder layers, each token attends to all tokens in the *encoded source sequence*. This allows the decoder to focus on relevant parts of the source when generating the target.
* **Location in Code:** Implemented in a `MultiHeadAttention` module, used by `EncoderLayer` and `DecoderLayer`.

### 3.4. Position-wise Feedforward Networks (FFN)

* **Purpose:** After the attention mechanism, each position in the sequence (independently) passes through a two-layer feedforward network.
* **Structure:** A linear transformation, followed by a ReLU activation, followed by another linear transformation.
* **Location in Code:** Implemented in a `PositionwiseFeedforwardLayer` module.

### 3.5. Layer Normalization and Residual Connections

* **Residual Connections:** Each sub-layer (attention and FFN) in both the Encoder and Decoder is wrapped in a residual connection, meaning the input to the sub-layer is added to its output. This helps with gradient flow and training deeper networks.
    * Output = `LayerNorm(x + Sublayer(x))`
* **Layer Normalization:** Applied after the residual connection. It normalizes the inputs across the features for each sample independently, helping to stabilize training.
* **Location in Code:** `LayerNorm` modules applied within `EncoderLayer` and `DecoderLayer`.

### 3.6. Encoder Structure

The Encoder consists of `num_layers` identical Encoder layers.

* **Each Encoder Layer:**
    1.  Multi-Head Self-Attention
    2.  Add & LayerNorm
    3.  Position-wise Feedforward Network
    4.  Add & LayerNorm
* **Input:** Source token embeddings + positional encodings.
* **Output:** A sequence of continuous representations, where each representation captures context from the entire source sentence.

### 3.7. Decoder Structure

The Decoder consists of `num_layers` identical Decoder layers.

* **Each Decoder Layer:**
    1.  Masked Multi-Head Self-Attention
    2.  Add & LayerNorm
    3.  Multi-Head Encoder-Decoder Attention
    4.  Add & LayerNorm
    5.  Position-wise Feedforward Network
    6.  Add & LayerNorm
* **Input:** Target token embeddings + positional encodings (shifted right during training for teacher forcing).
* **Output:** For each output token position, a representation that combines information from the previous target tokens and the entire encoded source sequence. This representation is then passed to a final linear layer to predict the next token.

## 4. Final Output Layer

* **Purpose:** A linear layer that maps the Decoder's output representations to the vocabulary size. The output of this layer is typically passed through a softmax function (often implicitly handled by `CrossEntropyLoss`) to produce probability distributions over the vocabulary for each output position, indicating the likelihood of each possible next token.

## 5. Model Parameters (Configurable)

Key parameters for the Transformer model are typically defined in configuration files:

* `d_model`: Dimension of embedding vectors and internal representations.
* `n_heads`: Number of attention heads in Multi-Head Attention.
* `num_encoder_layers`: Number of stacked Encoder layers.
* `num_decoder_layers`: Number of stacked Decoder layers.
* `d_ff`: Dimension of the inner layer of the Position-wise Feedforward Network.
* `dropout`: Dropout rate applied throughout the model to prevent overfitting.
* `src_vocab_size`: Size of the source language vocabulary.
* `tgt_vocab_size`: Size of the target language vocabulary.

By configuring these parameters, the model's capacity and computational cost can be scaled to suit the dataset size and available computing resources.
