import logging
import os
import sys

def get_logger(name):
    return logging.getLogger(name)

_LOG_LEVEL_MAP = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL,
}

def setup_logging(log_dir_path, run_log_name, log_level_str="INFO"):
    """
    Sets up logging to console and a file.
    Args:
        log_dir_path (str): Directory where log files will be saved.
        run_log_name (str): Base name for the log file (e.g., 'training').
        log_level_str (str): Logging level as a string (e.g., 'INFO', 'DEBUG').
    """
    os.makedirs(log_dir_path, exist_ok=True)

    root_logger = logging.getLogger()
    
    # Calculate log_level FIRST, outside the 'if' block
    log_level = _LOG_LEVEL_MAP.get(log_level_str.upper(), logging.INFO)

    # Only add handlers if they haven't been added already
    if not root_logger.handlers:
        root_logger.setLevel(logging.DEBUG) # Set to DEBUG to allow all messages to pass to handlers initially

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(log_level) # Set level for console output
        root_logger.addHandler(console_handler)

        # File handler
        log_file_path = os.path.join(log_dir_path, f"{run_log_name}.log")
        file_handler = logging.FileHandler(log_file_path)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(log_level) # Set level for file output
        root_logger.addHandler(file_handler)
    
    # After handlers are potentially added, ensure the root logger's level
    # and all handlers' levels are set to the desired log_level.
    # This also handles cases where setup_logging is called again with a new level.
    root_logger.setLevel(log_level)
    for handler in root_logger.handlers:
        handler.setLevel(log_level)