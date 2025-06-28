# General utility functions like config loading, device management, and checkpointing.

import logging
import yaml
import torch
import os
from typing import Dict, Any
from src.utils.logger import get_logger

logger = get_logger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Loads a YAML configuration file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        Dict[str, Any]: Loaded configuration dictionary.
    """
    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found: {config_path}")
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
        # Return the key that corresponds to the file name without extension
        # e.g., for config/data_config.yaml, return config['data_config']
        # This assumes the YAML structure is a single top-level key matching the file.
        # If the YAML directly contains parameters, return the whole dict.
        config_name = os.path.splitext(os.path.basename(config_path))[0]
        return config.get(config_name, config) # Try to return specific section, else full dict
    except yaml.YAMLError as exc:
        logger.error(f"Error parsing YAML file {config_path}: {exc}")
        raise

def get_device() -> torch.device:
    """
    Determines and returns the appropriate PyTorch device (CUDA if available, else CPU).

    Returns:
        torch.device: The selected device.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device

def save_checkpoint(state: dict, filepath: str, is_best: bool = False, best_model_filename: str = "best_model.pt"):
    """
    Saves a model checkpoint.

    Args:
        state (dict): A dictionary containing model's state_dict, optimizer_state_dict, etc.
        filepath (str): The full path where the current checkpoint will be saved.
        is_best (bool): If True, also saves this checkpoint as the 'best_model.pt'.
        best_model_filename (str): The filename for the best model (e.g., "best_model.pt").
    """
    try:
        torch.save(state, filepath)
        logger.debug(f"Checkpoint saved to {filepath}") # Use debug for frequent saves

        if is_best:
            # Construct path to the 'best_model.pt' in the same directory as filepath
            best_filepath = os.path.join(os.path.dirname(filepath), best_model_filename)
            torch.save(state, best_filepath)
            logger.info(f"New best model saved to {best_filepath}")
    except Exception as e:
        logger.error(f"Failed to save checkpoint to {filepath}: {e}")

def load_checkpoint(filepath: str, device: torch.device) -> Dict[str, Any]:
    """
    Loads a model checkpoint.

    Args:
        filepath (str): Path to the checkpoint file.
        device (torch.device): Device to load the checkpoint onto.

    Returns:
        Dict[str, Any]: Loaded checkpoint state dictionary.
    """
    if not os.path.exists(filepath):
        logger.error(f"Checkpoint file not found: {filepath}")
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")
    try:
        checkpoint = torch.load(filepath, map_location=device)
        logger.info(f"Checkpoint loaded from {filepath}")
        return checkpoint
    except Exception as e:
        logger.error(f"Error loading checkpoint from {filepath}: {e}")
        raise

def setup_logging(log_dir: str, run_name: str):
    """Sets up logging to console and a file."""
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, f"{run_name}.log")

    # Clear existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler()
        ]
    )
    # Suppress logging from external libraries if they are too verbose
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('wandb').setLevel(logging.ERROR) # Lower wandb verbosity in logs
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('datasets').setLevel(logging.WARNING)

    # Get the root logger
    logger = logging.getLogger()
    logger.info(f"Logging setup complete. Log file: {log_file_path}")
    return logger
