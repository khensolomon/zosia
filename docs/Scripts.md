# Scripts Overview

This document provides an overview of the shell scripts used within the Zosia Neural Machine Translation (NMT) project. These scripts automate various stages of the NMT pipeline, from data preparation to model training and text translation.

## What are Shell Scripts?

Shell scripts are plain text files containing a sequence of commands that an operating system's command-line interpreter (shell, e.g., Bash, PowerShell) can execute. In this project, they serve as convenient wrappers to automate complex multi-step processes, ensuring consistency and reproducibility.

## How to Run These Scripts

1. Navigate to the Project Root: All scripts should be executed from the project's main directory (where src, config, scripts folders reside).

    ```bash
    cd /path/to/zosia
    ```

    (Replace `/path/to/zosia` with the actual path to the project directory.)

2. Make Executable (Linux/macOS only): On Linux and macOS, scripts require execute permissions. Perform this step once for each script:

    ```bash
    chmod +x scripts/preprocess_data.sh
    chmod +x scripts/run_training.sh
    chmod +x scripts/translate_text.sh
    ```

3. Execute the Script:
   * On Linux / macOS (Bash/Zsh):

      ```bash
      ./scripts/script_name.sh [arguments]
      ```

   * On Windows (Git Bash / WSL):

      ```bash
      ./scripts/script_name.sh [arguments]
      ```

   * On Windows (PowerShell / Command Prompt): Requires sh or bash to be installed and in the system's PATH.

      ```bash
      sh scripts/script_name.sh [arguments]
      # or
      bash scripts/script_name.sh [arguments]
      ```

Important Windows Note for `run_training.sh`: Symbolic link creation on Windows may require the terminal to be run "As Administrator" or "Developer Mode" to be enabled in Windows settings due to operating system security policies.

## Script Details

### 1. `scripts/preprocess_data.sh`

* Purpose: Automates the data preprocessing pipeline.
* What it does:
* Trains a SentencePiece tokenizer based on the vocab_size and other settings in config/data_config.yaml. This creates the .model and .vocab files for the tokenizer.
* Tokenizes the raw source and target language text files into numerical ID sequences.
* Saves the tokenized data into PyTorch-compatible .pt files, ready for efficient loading by data loaders during training.
* How to run:

```bash
./scripts/preprocess_data.sh
```

* Benefits: Ensures consistent tokenizer training and data preparation, crucial for reproducibility and avoiding errors.

### 2. `scripts/run_training.sh`

Purpose: Initiates the model training process.

What it does:

* Reads model, data, and training hyperparameters from config/model_config.yaml, config/data_config.yaml, and config/training_config.yaml.
* Initializes the NMT Transformer model, optimizer, and learning rate scheduler.
* Sets up experiment directories (experiments/nmt_run_YYYYMMDD_HHMMSS).
* Creates or updates a symbolic link named experiments/latest_run to point to the newly started experiment directory, providing easy access to the most recent outputs.
* Starts the training loop, including logging and checkpointing.

How to run:

```bash
./scripts/run_training.sh
```

Benefits: Simplifies starting training, manages experiment organization automatically, and provides a clear entry point for reproducibility.

### 3. `scripts/translate_text.sh`

Purpose: Executes the translation of a given input sentence using the trained model.

What it does:

Takes a sentence as a command-line argument.

Loads the `best_model.pt` checkpoint from the `experiments/latest_run/checkpoints/` path (leveraging the symbolic link created by `run_training.sh`).

Utilizes the loaded model and tokenizer to translate the provided sentence.

Prints the translated output.

How to run:

```bash
./scripts/translate_text.sh "Your sentence goes here."
```

Note: Enclose the sentence in double quotes if it contains spaces.

Benefits: Enables quick testing and demonstration of the trained model without needing to manually specify long model paths.

## Overall Advantages of Using Scripts

Automation: Reduce manual steps and potential for human error.

Consistency: Guarantee the same sequence of operations and configurations are used every time.

Reproducibility: Facilitate sharing workflows, allowing others to easily replicate results.

Streamlined Workflow: Simplify complex multi-stage pipelines into single, manageable commands.

Experiment Management: Integrate with automatic experiment directory creation and easy access to latest results.
