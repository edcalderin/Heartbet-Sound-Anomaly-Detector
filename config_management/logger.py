from datetime import datetime
import logging
from logging import handlers
from pathlib import Path
import sys

def get_logger(
    module_name: str,
    log_location: str = '/tmp/logs',
    log_format: str = '%Y%m%d%H%M%S',
    logger_level: int = logging.INFO
):
    '''
    Custom logger to use in the app

    Parameters:
    ----------
    - app_name: name of the application or floe that is running
    - logger_level: logger level - CRITICAL=50, ERROR=40, WARNING=30, INFO=20, DEBUG=10, NOTSET=0
    '''
    #existing_logger = logging.getLogger(module_name)
    #if existing_logger.handlers:
    #    # Logger already configured, return the existing logger
    #    return existing_logger

    log_save = Path(log_location).joinpath(f'{datetime.now().strftime(log_format)}_{module_name}.log')

    logger = None
    try:
        logger = logging.getLogger(module_name)
        logger.setLevel(logger_level)

        format = logging.Formatter(
            '%(asctime)s - [%(levelname)s] - [%(name)s]: %(message)s')
        loginStreamHandler = logging.StreamHandler()
        loginStreamHandler.setFormatter(format)
        logger.addHandler(loginStreamHandler)

        fileHandler = handlers.RotatingFileHandler(log_save, maxBytes = (1048576 * 5), backupCount = 7)
        fileHandler.setFormatter(format)
        logger.addHandler(fileHandler)

    except Exception:
        logger = None

    return logger