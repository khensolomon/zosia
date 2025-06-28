# Project Overview: Zotranslate NMT System

This document provides a high-level overview of the Zotranslate Neural Machine Translation (NMT) project, outlining its purpose, core components, and general workflow.

## 1. Purpose

The Zotranslate NMT project aims to develop a machine translation system capable of translating text between the Zomi language and English. The primary objective is to build a robust and performant translation model, leveraging modern deep learning architectures.

## 2. Core Components

The project is structured into several logical components, each responsible for a specific part of the NMT pipeline:

* **Data (`data/`):** Contains all raw, intermediate, and processed data necessary for training and evaluation.
    * `data/raw/`: Stores raw parallel and monolingual text files. Includes a YAML catalog (`base_training.yaml`) to define data sources.
    * `data/processed/`: Stores tokenized and pre-processed numerical data (PyTorch `.pt` files) ready for model training.
    * `data/vocab/`: Houses the trained SentencePiece tokenizer model (`.model`) and its associated vocabulary file (`.vocab`).
* **Source Code (`src/`):** Contains the Python modules that implement the project's logic.
    * `src/data/`: Modules for data loading, cleaning, tokenization, and dataset creation (e.g., `make_dataset.py`, `dataset_utils.py`).
    * `src/models/`: Definitions of the NMT model architecture, including Encoder, Decoder, Attention mechanisms, and the main Transformer model.
    * `src/training/`: Scripts and utilities for setting up and executing the training loop, including optimization, loss calculation, and evaluation during training (e.g., `trainer.py`).
    * `src/utils/`: General utility functions and helper modules (e.g., `general_utils.py` for configuration loading, logging setup).
    * `src/inference/`: (Planned/To be implemented) Modules for performing inference (translation) on new text using a trained model.
* **Configuration (`config/`):** Stores YAML configuration files that control various aspects of the project, from data paths to model hyperparameters and training settings.
* **Experiments (`experiments/`):** Stores logs, checkpoints of trained models, and other outputs from training runs. Each run typically gets its own timestamped directory.

## 3. General Workflow

The typical workflow for developing and utilizing the NMT system involves the following steps:

1.  **Project Setup:** Clone the repository and set up a Python virtual environment with all required dependencies.
2.  **Data Acquisition & Organization:** Raw parallel and monolingual text data is gathered and organized in the `data/raw/` directory, with its structure defined in `base_training.yaml`.
3.  **Data Preparation:** The `make_dataset.py` script is executed to clean, tokenize, and numerically encode the raw text, saving the results as PyTorch tensors (`.pt` files) in `data/processed/`. This step also trains and saves the SentencePiece tokenizer.
4.  **Model Training:** The `trainer.py` script initiates the training process. The model learns to translate by optimizing its parameters based on the prepared parallel data. Training progress is monitored via logging and tools like Weights & Biases (W&B). Checkpoints of the model's state are saved periodically.
5.  **Evaluation:** During training, the model's performance is evaluated on a validation set. After training, a final evaluation is performed on a dedicated test set.
6.  **Inference/Deployment:** (Future Step) The best-performing trained model and the tokenizer are used to translate new, unseen text on a deployment environment.

This structured approach ensures modularity, reproducibility, and ease of management for the NMT development lifecycle.