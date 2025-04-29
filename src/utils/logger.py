import logging
import os

def get_logger(log_dir=None, log_level=logging.INFO):
    """
    Sets up a logger with console and optional file logging.

    Args:
        log_dir (str, optional): Directory to save log files. If None, only console logging is enabled.
        log_level (int, optional): Logging level. Default is logging.INFO.

    Returns:
        logger (logging.Logger): Configured logger.
    """
    logger = logging.getLogger("voice-cloning")
    logger.setLevel(log_level)

    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Optional file handler
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(log_dir, 'training.log'))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
