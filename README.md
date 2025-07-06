# ZoSia: A Zolai-English Translation Project

This project contains a complete pipeline for training a neural machine translation (NMT) model for Zolai and English, based on a Seq2Seq architecture with an attention mechanism. It also includes several command-line tools for translation, suggestions, and language detection.

## Features

- **Bidirectional Training:** Train separate, optimized models for both Zolai-to-English (`zo-en`) and English-to-Zolai (`en-zo`) translation.
- **Attention Mechanism:** Utilizes a Bahdanau-style attention mechanism for improved handling of long sentences.
- **Config-Driven:** All settings, paths, and hyperparameters are managed via YAML configuration files in the `/config` directory.
- **Automatic Language Detection:** User-facing tools can automatically detect the input language (`zo` or `en`).
- **Command-Line Tools:**
    - `translate.py`: Translate text with auto-detection.
    - `suggest.py`: Get monolingual autocomplete suggestions.
    - `detector.py`: A standalone language identification tool.

---

## 1. Project Setup

Follow these steps to set up your local development environment.

### Step 1: Clone the Repository

First, clone the project to your local machine.

```bash
git clone <your-repository-url>
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

Before training the models, you must generate the language profiles needed for the automatic language detector.

Run the following command from the project's root directory:

```bash
python -m scripts.build_profiles
```

This will read the raw text data from `./data/parallel_base/` and create `en.profile.json` and `zo.profile.json` inside the `./data/locale/` directory. You only need to do this once, or whenever your raw data changes significantly.

---

## 3. Training the Models

You need to train two separate models for bidirectional translation.

### Train the Zolai-to-English Model

```bash
python -m zo.sia.main --source zo --target en
```

### Train the English-to-Zolai Model

```bash
python -m zo.sia.main --source en --target zo
```

After each command completes, a `.pth` checkpoint file will be saved in the `./experiments/` directory (e.g., `ZoSia_zo-en_checkpoint.pth`).

---

## 4. Using the Command-Line Tools

Once the models are trained, you can use the following tools.

### `translate.py`

Translates text. It will auto-detect the language if not specified.

```bash
# Auto-detect language
python -m zo.sia.translate --text "kei hong paita"
python -m zo.sia.translate --text "how are you"

# Manually specify direction
python -m zo.sia.translate --source en --target zo --text "hello world"

# Start interactive mode
python -m zo.sia.translate
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

# Run the built-in demonstration
python -m zo.sia.detector
