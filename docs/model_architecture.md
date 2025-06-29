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
    1. Multi-Head Self-Attention
    2. Add & LayerNorm
    3. Position-wise Feedforward Network
    4. Add & LayerNorm
* **Input:** Source token embeddings + positional encodings.
* **Output:** A sequence of continuous representations, where each representation captures context from the entire source sentence.

### 3.7. Decoder Structure

The Decoder consists of `num_layers` identical Decoder layers.

* **Each Decoder Layer:**
  1. Masked Multi-Head Self-Attention
  2. Add & LayerNorm
  3. Multi-Head Encoder-Decoder Attention
  4. Add & LayerNorm
  5. Position-wise Feedforward Network
  6. Add & LayerNorm
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
