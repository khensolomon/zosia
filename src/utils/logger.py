# A simple logging utility.

import logging
import os

def get_logger(name: str, log_level: str = "INFO") -> logging.Logger:
    """
    Configures and returns a logger instance.

    Args:
        name (str): The name of the logger (usually __name__).
        log_level (str): The minimum logging level (e.g., "INFO", "DEBUG", "WARNING").

    Returns:
        logging.Logger: Configured logger object.
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Prevent duplicate handlers from being added if get_logger is called multiple times
    if not logger.handlers:
        # Console handler
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger

def setup_logging(log_file_path: str, log_level: str = "INFO"):
    """
    Sets up the root logger to log to a file and console.
    Call this once at the start of your main script.

    Args:
        log_file_path (str): Full path to the log file.
        log_level (str): Minimum logging level.
    """
    # Ensure the directory exists for the log file
    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Clear existing handlers to prevent duplicate output if called multiple times
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # File handler
    file_handler = logging.FileHandler(log_file_path)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    # Console handler (can be configured differently if needed)
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    root_logger.info(f"Logging initialized. Output will be saved to {log_file_path}")