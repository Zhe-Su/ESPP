import logging


def get_logger(name='trainer'):
    """
    Get a logger instance.

    Args:
        name (str): Logger name
        
    Returns:
        logging.Logger: Configured logger instance
    """
    return logging.getLogger(name)
