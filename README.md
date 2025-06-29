# ZOSIA: ZO-EN Neural Machine Translation (NMT) System

## Project Goal

To develop a robust and customizable neural machine translation system capable of accurately translating text between Zo (ZO) and English (EN) in both directions (ZO -> EN and EN -> ZO). This project focuses on building a high-quality NMT system for a low-resource language like Zo, emphasizing modularity, customizability, and clear documentation.

## Features

* **Modular Architecture:** Utilizes a standard Transformer-based sequence-to-sequence model.
* **Customizable Training:** Configurable hyperparameters for data processing, model architecture, and training routines via YAML files.
* **Subword Tokenization:** Employs SentencePiece for efficient and robust tokenization suitable for various languages.
* **Experiment Tracking:** Integrates with Weights & Biases (W&B) for comprehensive experiment logging and visualization.
* **Comprehensive Documentation:** Detailed guides for setup, data preparation, and project structure are provided in the `docs/` directory.

## Table of Contents

- [ZOSIA: ZO-EN Neural Machine Translation (NMT) System](#zosia-zo-en-neural-machine-translation-nmt-system)
  - [Project Goal](#project-goal)
  - [Features](#features)
  - [Table of Contents](#table-of-contents)
  - [1. Getting Started](#1-getting-started)
    - [1.1. Setup and Environment](#11-setup-and-environment)
    - [1.2. Data Preparation](#12-data-preparation)
  - [2. Project Structure \& Model Architecture](#2-project-structure--model-architecture)
    - [2.1. Directory Structure Overview](#21-directory-structure-overview)
    - [2.2. Model Architecture Details](#22-model-architecture-details)
  - [3. Usage](#3-usage)
    - [3.1. Training the Model](#31-training-the-model)
    - [3.2. Running Inference (Translation)](#32-running-inference-translation)
  - [4. Customization](#4-customization)
  - [5. Key Technologies](#5-key-technologies)
  - [6. Contributing](#6-contributing)
  - [7. License](#7-license)
  - [8. Contact](#8-contact)

## 1. Getting Started

This section provides a quick guide to setting up the project and preparing data for training. For detailed instructions, refer to the dedicated documentation files.

### 1.1. Setup and Environment

Information on prerequisites, repository cloning, virtual environment creation, and dependency installation is available in the [Setup Guide](docs/setup_guide.md).

### 1.2. Data Preparation

Instructions for collecting, cleaning, formatting, and processing Zo-English parallel and monolingual data are provided in the [Data Preparation Guide](docs/data_preparation_guide.md). It is crucial that data is correctly prepared before training.

## 2. Project Structure & Model Architecture

### 2.1. Directory Structure Overview

A high-level overview of the project's modular and organized directory structure can be found in the [Project Overview](docs/project_overview.md).

### 2.2. Model Architecture Details

Detailed information about the Transformer model architecture, its components (Encoder, Decoder, Attention mechanisms), and their implementation is available in the [Model Architecture documentation](docs/model_architecture.md).

## 3. Usage

### 3.1. Training the Model

After setting up the environment and preparing the data, model training can be initiated.

```bash
python -m src.train.trainer
```

For detailed configuration options and advanced training parameters, refer to the relevant sections in the config/ directory.

### 3.2. Running Inference (Translation)

(Detailed instructions will be added here once src/inference/translator.py is implemented and documented.)

Example inference command:

```bash
python -m src.inference.translator --model_path experiments/latest_run/checkpoints/best_model.pt --text "How are you"

python -m src.inference.translator --model_path experiments/nmt_run_20250628_114934/checkpoints/best_model.pt --text "Dam maw"

python -m src.translate.translator --checkpoint_path experiments/nmt_run_20250628_114934/checkpoints/best_model.pt --sentence "Dam maw"

```

## 4. Customization

The system's behavior, including data processing, model hyperparameters, and training routines, can be customized by modifying the YAML configuration files located in the config/ directory. Refer to the documentation within those files for specific parameters.

## 5. Key Technologies

* Python: Primary programming language.
* PyTorch: Deep learning framework.
* SentencePiece: For efficient subword (BPE/Unigram) tokenization.
* Weights & Biases (W&B): For comprehensive experiment tracking and visualization.
* Pandas & NumPy: For data manipulation and numerical operations.
* Tqdm: For displaying progress bars during long-running operations.

## 6. Contributing

Contributions are welcome! Please refer to CONTRIBUTING.md (to be created) for guidelines.

## 7. License

This project is licensed under the MIT License - see the LICENSE file for details.

## 8. Contact

For questions or collaborations, please reach out to [issues](https://github.com/khensolomon/zosia/issues).
