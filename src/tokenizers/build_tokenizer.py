import os
import csv
import sentencepiece as spm
import yaml
from src.utils.general_utils import get_logger

logger = get_logger(__name__)

# --- START OF get_monolingual_files_for_tokenizer FUNCTION ---
# This entire function definition should be placed BEFORE build_tokenizer
def get_monolingual_files_for_tokenizer(base_data_path: str):
    """
    Reads index.csv and returns a list of full paths to monolingual files
    marked for 'process', ignoring lines starting with '#'.
    Assumes structure: <base_data_path>/monolingual/zo/
    """
    monolingual_zo_dir = os.path.join(base_data_path, "monolingual", "zo")
    context_csv_path = os.path.join(monolingual_zo_dir, "index.csv")

    if not os.path.exists(monolingual_zo_dir):
        logger.error(f"Monolingual Zolai directory not found at: {monolingual_zo_dir}")
        raise FileNotFoundError(f"Monolingual Zolai directory not found at: {monolingual_zo_dir}")
    if not os.path.exists(context_csv_path):
        logger.error(f"index.csv not found at: {context_csv_path}")
        raise FileNotFoundError(f"index.csv not found at: {context_csv_path}")

    files_to_process = []
    with open(context_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for line_num, row in enumerate(reader):
            if not row: # Skip empty lines
                continue
            
            # Check if the first element (filename) starts with '#' after stripping whitespace
            if row[0].strip().startswith('#'):
                logger.info(f"Skipping commented line in index.csv: {row}")
                continue

            if len(row) < 2:
                logger.warning(f"Invalid line in index.csv at line {line_num + 1}: {row}. Skipping.")
                continue

            filename = row[0].strip()
            action = row[1].strip().lower()

            if action == "process":
                full_file_path = os.path.join(monolingual_zo_dir, filename)
                if not os.path.exists(full_file_path):
                    logger.warning(f"File specified in index.csv not found: {full_file_path}. Skipping.")
                else:
                    files_to_process.append(full_file_path)

    return files_to_process
# --- END OF get_monolingual_files_for_tokenizer FUNCTION ---


# --- START OF build_tokenizer FUNCTION ---
def build_tokenizer(config_path: str = 'config/tokenizer_config.yaml', base_data_path: str = 'data'):
    # Get the absolute path of the directory containing THIS SCRIPT (src/tokenizers)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Go up two levels to reach the ZoTranslate root directory
    # src/tokenizers -> src -> ZoTranslate/
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    
    # Now, construct the absolute path to the 'data' folder
    absolute_data_path = os.path.join(project_root, base_data_path)
    logger.info(f"Calculated absolute base data path: {absolute_data_path}")
    
    logger.info(f"Loading tokenizer configuration from {config_path}")
    # This config_path is still relative to your current working directory (C:\dev\zosia)
    # If you wanted it relative to project_root, you'd do:
    # absolute_config_path = os.path.join(project_root, config_path)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    vocab_size = config['vocab_size']
    model_type = config['model_type']
    model_prefix = config['model_prefix']

    # Define where the tokenizer model files will be saved using the absolute path
    tokenizer_output_dir = os.path.join(absolute_data_path, "tokenizers")
    os.makedirs(tokenizer_output_dir, exist_ok=True) # Ensure the directory exists
    model_path = os.path.join(tokenizer_output_dir, model_prefix)

    logger.info(f"Attempting to save tokenizer model to: {model_path}.model (Absolute Path Debug)")
    
    # Pass the absolute path to get_monolingual_files_for_tokenizer
    input_files = get_monolingual_files_for_tokenizer(absolute_data_path)
    if not input_files:
        logger.error("No input files found for tokenizer training. Please check index.csv and file paths.")
        raise ValueError("No input files for tokenizer training.")

    # Convert list of file paths to a comma-separated string for SentencePieceTrainer
    input_argument = ','.join(input_files)
    
    logger.info(f"Training SentencePiece tokenizer with input files: {input_argument}")
    logger.info(f"Vocab size: {vocab_size}, Model type: {model_type}, Output prefix: {model_path}")

    # SentencePiece training command
    spm.SentencePieceTrainer.train(
        input=input_argument, 
        model_prefix=model_path, 
        vocab_size=vocab_size, 
        model_type=model_type,
        character_coverage=0.9995, # Common setting to cover most characters
        split_by_unicode_script=True # Good for multilingual data, but fine for single language too
    )

    logger.info(f"Tokenizer training complete. Model saved to {model_path}.model and {model_path}.vocab")

if __name__ == "__main__":
    # Example usage: run this script directly
    build_tokenizer(
        config_path='config/tokenizer_config.yaml',
        base_data_path='data' # This argument 'data' is interpreted relative to project_root
    )