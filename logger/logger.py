import logging
import os
import inspect


def setup_logger(log_file_path):
    """
    Sets up a logger that writes to a file and includes file information.

    Args:
        log_file_path (str): The path to the log file.

    Returns:
        logging.Logger: The configured logger.
    """
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    # Create a file handler that writes to the specified file
    file_handler = logging.FileHandler(
        log_file_path, mode='a', encoding='utf-8')  # Append mode

    # Define the log message format, including filename, line number, and function name
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s() - %(message)s')
    file_handler.setFormatter(formatter)

    # Create a logger
    # Or use a specific name if you have a module-specific logger
    logger = logging.getLogger(__name__)
    # Set the logging level (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL)
    logger.setLevel(logging.DEBUG)

    # Add the file handler to the logger
    logger.addHandler(file_handler)

    return logger


def get_logger(log_file_path="my_app.log"):
    """
    Retrieves the logger, setting it up if it doesn't exist.  This ensures
    the logger is initialized only once.

    Returns:
        logging.Logger: The logger.
    """
    global _logger  # Use a global variable to store the logger instance
    if '_logger' not in globals():
        _logger = setup_logger(log_file_path)
    return _logger
