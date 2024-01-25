import logging
import sys
from pathlib import Path
import yaml
from config_management.logger import get_logger

logger = get_logger(module_name = 'config-reader', logger_level = logging.INFO, log_location = 'logs')

current_directory = Path(__file__).parent

class ConfigReader:
    @staticmethod
    def read_params(path: Path):
        try:
            with open(path) as file:
                return yaml.safe_load(file)

        except yaml.YAMLError as e:
            logger.error(f'Invalid or corrupted yaml file: {e}')
            exit(-1)
        except Exception as e:
            logger.error(f'Error by reading yaml file: {e}')
            exit(-1)

config_params = ConfigReader.read_params(current_directory.joinpath('configuration.yaml'))