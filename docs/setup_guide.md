# Local Machine Setup Guide (Training & Testing Environment)

This guide outlines the steps to set up the NMT project on a local machine, prepare the data, and run training and evaluation.

## 1. Prerequisites

Before starting, ensure the following software is installed on the local machine:

* **Git:** For cloning the project repository.
  * [Download Git](https://git-scm.com/downloads)
* **Python (3.8+ recommended):** The programming language for the project.
  * [Download Python](https://www.python.org/downloads/) (On Windows, consider checking "Add Python to PATH" during installation)
* **pip:** Python's package installer (typically included with Python installations).

## 2. Project Setup

1. **Clone the Repository:**
    Open a terminal or command prompt and clone the project's Git repository.

    ```bash
    git clone https://github.com/khensolomon/zosia.git
    ```

2. **Navigate into the Project Directory:**
    Change the current directory to the root of the cloned project.

    ```bash
    cd zosia # Replace 'zosia' with the actual name of the cloned project folder.
    ```

## 3. Python Environment Setup

Using a virtual environment is highly recommended to manage project dependencies and prevent conflicts with other Python projects.

1. **Create a Virtual Environment:**

    ```bash
    python -m venv venv
    ```

    (This command creates a folder named `venv` within the project directory to house the virtual environment.)

2. **Activate the Virtual Environment:**
    * **On Windows:**

        ```bash
        .\venv\Scripts\activate
        ```

    * **On macOS/Linux:**

        ```bash
        source venv/Scripts/activate
        ```

    (The command prompt should display `(venv)` at the beginning, indicating the environment is active.)

3. **Install Project Dependencies:**
    Install all required Python packages listed in the `requirements.txt` file.

    ```bash
    pip install -r requirements.txt
    ```

## 4. Data Preparation

Before training, the raw text data must be tokenized and processed into a format suitable for the model.

1. **Place Raw Data:**
    Ensure raw parallel text files (e.g., `train.zo`, `train.en`, `val.zo`, `val.en`, `test.zo`, `test.en`) are placed within the `data/raw/` directory of the project. This also includes any monolingual data specified in the configuration.

    * **Important:** Verify that the `data/raw/base_training.yaml` configuration file correctly specifies the paths to these raw data files and defines the source and target language extensions (e.g., `.zo`, `.en`). Also, check `data/monolingual/zo/index.csv` for monolingual data sources.

2. **Run Data Processing Script:**
    This script will train the SentencePiece tokenizer and subsequently tokenize the raw data, saving the processed token IDs as `.pt` files in the `data/processed/` directory.

    ```bash
    python -m src.data.make_dataset
    ```

    * **Expected Output:** This command should generate `zosia_sp.model` in `data/vocab/` and files like `train_token_ids.zo.pt`, `train_token_ids.en.pt`, `val_token_ids.zo.pt`, etc., in `data/processed/`.

## 5. Configuration

Review the main configuration files (e.g., `config/data_config.yaml`, `config/model_config.yaml`, `config/training_config.yaml` or a combined `config.yaml`). These files contain important hyperparameters for training, such as:

* **Data Parameters:** Maximum sequence length, vocabulary size.
* **Model Parameters:** Model dimensions (`d_model`), number of encoder/decoder layers, number of attention heads, dropout rates.
* **Training Parameters:** Learning rate, batch size, number of epochs, optimizer type, loss function details.

Adjust these parameters as necessary, particularly after increasing the dataset size or if performance issues arise.

## 6. Running Training

Once the data is prepared and dependencies are installed, the training process can be initiated.

1. **Start Training:**

    ```bash
    python -m src.train.trainer
    ```

    * **Logging:** The project is configured to use Weights & Biases (W&B) for tracking. It will likely operate in `offline` mode by default. An option to sync to the cloud may be prompted by W&B.
    * **Monitoring:** Terminal output and W&B logs (if synced) will display training progress, including loss, perplexity, and BLEU score per epoch.

## 7. Running Evaluation/Testing

Evaluation on the validation set is typically integrated into the training loop. Final evaluation on the test set is often performed after training concludes, utilizing the best-performing model checkpoint.

1. **Automatic Validation:**
    During the `python -m src.train.trainer` execution, evaluation on the validation set will occur periodically (usually at the end of each epoch).

2. **Final Test Set Evaluation (If Separate Script):**
    If a dedicated script exists for final testing (e.g., `src/inference/evaluate.py` or similar), it would typically be run as follows:

    ```bash
    python -m src.inference.evaluate --model_path experiments/nmt_run_YYYYMMDD_HHMMSS/checkpoints/best_model.pt
    ```

    (Adjust the path to the best model checkpoint and the script name as necessary.)

## Troubleshooting Tips

* **"No module named..." errors:** Ensure the virtual environment is activated and `pip install -r requirements.txt` was executed successfully.
* **File Not Found errors:** Double-check all file paths specified in configuration files and verify that `make_dataset.py` successfully generated all expected `.pt` files in `data/processed/` and the SentencePiece model in `data/vocab/`.
* **"N/A samples" in DataLoader summary:** This indicates an issue with data loading. Review the `src/data/dataset_utils.py` module for any errors in dataset initialization or loading logic.
* **Low BLEU / High PPL:** This is frequently a symptom of **insufficient training data**. NMT models, particularly Transformers, require substantial datasets (minimum tens to hundreds of thousands of parallel sentences) to learn effectively.
