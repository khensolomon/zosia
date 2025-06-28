# NMT Model Deployment: Essential Files and Understanding "Trained Data"

This document clarifies which files are essential for deploying a Neural Machine Translation (NMT) model for inference (translation) and defines what "trained data" refers to in this context.

## 1. Files Required on a Deployment Machine for NMT

A deployment machine's primary purpose is to demonstrate results (perform inference) and typically does not require the extensive components used for training, data generation, or compilation.

Here are the key categories of files generally needed for NMT model deployment:

### 1.1. The Trained Model File (Model Weights / Checkpoint)
* **What it is:** This is the most critical component. It contains all the numerical parameters (weights and biases) that the Transformer model learned during its training process. It is often referred to as a "checkpoint file" or a "model's state dictionary."
* **Purpose on Deployment:** When this file is loaded into the model's architecture, the model becomes "trained" and capable of performing translations.
* **Typical Location/Filename:** Often found in an `experiments/<run_name>/checkpoints/` directory. The file named `best_model.pt` (or `.pth`) typically represents the model that achieved the best performance during training. For example, `experiments/nmt_run_20250628_052650/checkpoints/best_model.pt`.

### 1.2. The Tokenization Model File (SentencePiece Model)
* **What it is:** An NMT model processes numerical token IDs rather than raw text. This file contains the vocabulary and rules necessary to convert raw input sentences into token IDs for the model (tokenization) and to convert the model's output token IDs back into readable text (detokenization). SentencePiece is a common tool for this.
* **Purpose on Deployment:** Essential for preparing any new input text for translation and converting the model's numerical output into a human-readable translated sentence.
* **Typical Location/Filename:** For instance, `data/vocab/zosia_sp.model`.

### 1.3. The Model Architecture Definition (Python Code)
* **What it is:** This refers to the Python source code that defines the structural blueprint of the Transformer model and its sub-components (e.g., `Encoder`, `Decoder`, `MultiHeadAttention`, `TokenEmbedding`, `PositionalEncoding`). The deployment environment requires this code to construct the empty framework of the model before its trained weights can be loaded.
* **Purpose on Deployment:** To instantiate the `Transformer` class and all its nested layers, thus creating the computational graph where the loaded weights will reside.
* **Typical Location/Filename Examples:** Python files within the `src/models/` directory, such as `src/models/transformer.py`, `src/models/encoder_decoder.py`, `src/models/attention.py`, `src/models/embeddings.py`, `src/models/utils.py`.

### 1.4. Inference Script / Utility Code
* **What it is:** A dedicated Python script or a collection of functions designed to manage the entire translation workflow for deployment. This code typically:
    * Loads the tokenizer model.
    * Instantiates the Transformer model architecture.
    * Loads the trained model weights (`best_model.pt`) into the model instance.
    * Implements the translation logic (e.g., greedy decoding or beam search).
    * Handles preprocessing of new input sentences (tokenization, adding special tokens, padding).
    * Manages post-processing of the model's output (detokenization, removal of special tokens).
* **Purpose on Deployment:** To provide an interface or function to accept a raw input string and return its translated counterpart.
* **Typical Location/Filename Examples:** A new file like `src/inference/predict.py` or a function within a `src/utils/inference_utils.py`.

### 1.5. Configuration Files (Optional but Recommended)
* **What it is:** Plain text files (e.g., YAML) used to store essential model hyperparameters (e.g., `d_model`, `n_heads`, `num_layers`), paths to the tokenizer and model weights, maximum sequence length, and other settings.
* **Purpose on Deployment:** Allows for adjustments to deployment-specific settings or model parameters without direct modification of the code.
* **Typical Location/Filename Examples:** `config/inference_config.yaml` or a relevant section within a main `config.yaml`.

---

## 2. Files NOT Typically Required on a Deployment Machine

* **Raw Data (`.txt` files):** Original text files used solely for dataset creation.
* **Processed Data (`.pt` token ID files in `data/processed/`):** The tokenized training, validation, and test datasets.
* **Training Scripts (`src/training/trainer.py`):** Code responsible for executing the training loop.
* **Data Generation Scripts (`src/data/make_dataset.py`):** Code that converts raw text into tokenized `.pt` files.
* **Experiment Tracking Logs or Setup (e.g., Wandb):** Specific to development and experiment tracking.
* **Optimizer State or Learning Rate Scheduler State:** Relevant only for resuming training from a checkpoint.
* **Development-only Libraries or Tools:** Any libraries primarily used during the development or training phase and not for inference.

---

## 3. What is "Trained Data" in this Context?

When discussing "trained data" as the *result* of the training process that the model utilizes for prediction, the term refers to the **trained model weights**.

* **The "trained data" is effectively the `best_model.pt` file.**
* This file does not contain raw sentences or token lists; rather, it encapsulates the numerical representation of the **knowledge and patterns** that the model has learned from processing its extensive training dataset. This acquired knowledge is encoded within the model's parameters, which are then saved in this `.pt` file.

In essence, raw data is fed into the training process, and the `best_model.pt` file emerges as the distilled "knowledge" ready for use in deployment.