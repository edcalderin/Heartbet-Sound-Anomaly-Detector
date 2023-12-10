import logging
import sys
from pathlib import Path
import yaml
from yaml.loader import SafeLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

current_directory = Path(__file__).parent

class ConfigReader:
    @staticmethod
    def read_params(path: Path):
        try:
            with open(path) as file:
                return yaml.load(file, Loader=SafeLoader)
            
        except yaml.YAMLError as e:
            logging.error(f'Invalid yaml file or corrupted yaml file: {e}')
            sys.exit()
        except Exception as e:
            logging.error(f'Error by reading yaml file: {e}')
            sys.exit()

config_params = ConfigReader.read_params(current_directory.joinpath('configuration.yaml'))