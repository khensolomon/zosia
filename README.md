# ZoSia: A Zolai-English Translation Project

This project contains a complete pipeline for training a neural machine translation (NMT) model for Zolai and English, based on a Seq2Seq architecture with an attention mechanism. It also includes several command-line tools for translation, suggestions, and language detection.

## Features

- **Bidirectional Training:** Train separate, optimized models for both Zolai-to-English (`zo-en`) and English-to-Zolai (`en-zo`) translation.
- **Back-Translation:** Augment training data with synthetic parallel data generated from monolingual text to improve model fluency.
- **Attention Mechanism:** Utilizes a Bahdanau-style attention mechanism for improved handling of long sentences.
- **Config-Driven:** All settings, paths, and hyperparameters are managed via YAML configuration files in the `/config` directory.
- **Automatic Language Detection:** User-facing tools can automatically detect the input language (`zo` or `en`).
- **Command-Line Tools:**
    - `translate.py`: Translate text with auto-detection.
    - `suggest.py`: Get monolingual autocomplete suggestions.
    - `detector.py`: A standalone language identification tool.
    - `analyze_model.py`: Inspect and analyze trained model checkpoints.

---

## 1. Project Setup

Follow these steps to set up the local development environment.

### Step 1: Clone the Repository

First, clone the project to a local machine.

```bash
git clone https://github.com/khensolomon/zosia
cd zosia
```

### Step 2: Create and Activate a Virtual Environment

It is highly recommended to use a virtual environment to manage project dependencies.

```bash
# Create the virtual environment
python -m venv venv

# Activate it
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

Install all required packages for development, including testing and analysis tools. This command installs everything from both `requirements.txt` and `requirements-dev.txt`.

```bash
pip install -r requirements-dev.txt
```

---

## 2. Initial Data Preparation

Before training the models, the language profiles needed for the automatic language detector must be generated.

Run the following command from the project's root directory:

```bash
python -m scripts.build_profiles
```

This will read the raw text data from `./data/parallel_base/` and create `en.profile.json` and `zo.profile.json` inside the `./data/locale/` directory. This only needs to be done once, or whenever the raw data changes significantly.

---

## 3. Training the Models

Two separate models need to be trained for bidirectional translation.

### Step 1: Train the Base Models

First, train the models using only the high-quality parallel data.

```bash
# Train the Zolai-to-English Model
python -m zo.sia.main --source zo --target en

# Train the English-to-Zolai Model
python -m zo.sia.main --source en --target zo
```

After each command completes, a `.pth` checkpoint file will be saved in the `./experiments/` directory.

### Step 2: Augment Data with Back-Translation (Optional, Recommended)

To improve model fluency, synthetic training data can be created from the monolingual text files.

First, run the back-translation script. This command uses the `zo-en` model to translate the monolingual Zolai data, creating new synthetic English data.

```bash
# This creates a new directory at ./data/synthetic/zo-en/
python -m scripts.back_translate --source zo --target en
```

Next, run it in the other direction to create synthetic Zolai data.

```bash
# This creates a new directory at ./data/synthetic/en-zo/
python -m scripts.back_translate --source en --target zo
```

### Step 3: Configure and Retrain with Augmented Data

Now, update `config/data.yaml` to include these new synthetic data sources. Add the `synthetic` paths to the `sources` list:

```yaml
# ./config/data.yaml
sources:
  - name: "parallel_base"
    path: "${paths.parallel_base}"

  # Add these new sources
  - name: "synthetic_zo-en"
    path: "${paths.root}/data/synthetic/zo-en"
    use_for_direction: "en-zo"
  - name: "synthetic_en-zo"
    path: "${paths.root}/data/synthetic/en-zo"
    use_for_direction: "zo-en"
    
  - name: "templates"
    path: "${paths.templates}"
# ...
```

Finally, run the training commands again. The data loader will now automatically include the synthetic data for the correct training direction, resulting in a "smarter" and more fluent model.

```bash
python -m zo.sia.main --source zo --target en
python -m zo.sia.main --source en --target zo
```

---

## 4. Using the Command-Line Tools

Once the models are trained, the following tools can be used.

### `translate.py`

Translates text. It will auto-detect the language if not specified.

```bash
# Auto-detect language
python -m zo.sia.translate --text "kei hong paita"

# Manually specify direction
python -m zo.sia.translate --source en --target zo --text "hello world"
```

### `suggest.py`

Provides monolingual autocomplete suggestions.

```bash
# Auto-detect language of the prefix
python -m zo.sia.suggest --text "how are"

# Manually specify language
python -m zo.sia.suggest --lang zo --text "hong paita"
```

### `detector.py`

A standalone tool to test the language detector.

```bash
# Run a specific test
python -m zo.sia.detector --text "zomi"
```

### `analyze_model.py`

Inspects a trained model checkpoint and prints a detailed report.

```bash
# Analyze the default zo->en model
python -m scripts.analyze_model --source zo --target en

# Analyze a specific model file by path
python -m scripts.analyze_model --model ./experiments/ZoSia_en-zo_checkpoint.pth
```

---

## 5. Running Tests

The project uses `pytest` for running tests. The test files are located in the `./tests/` directory.

### Running All Tests

To run the entire test suite, execute the following command from the project's root directory:

```bash
pytest
```

### Test Suite Breakdown

-   **`tests/test_full_pipeline.py` (Integration Test):** This test simulates the complete user workflow. It runs a short training session, generates a model checkpoint, and then uses that checkpoint with the `translate.py` and `suggest.py` scripts to ensure all parts of the system work together correctly.

-   **`tests/test_models.py` (Unit Test):** This test checks the individual components of the neural network architecture defined in `zo/sia/model.py`. It ensures that the tensor shapes and data flow within the Encoder and Decoder are correct.

-   **`tests/test_detector.py` (Unit Test):** This test verifies the accuracy of the `LanguageDetector`. It feeds it a series of known English and Zolai sentences and asserts that the detector classifies them correctly.
