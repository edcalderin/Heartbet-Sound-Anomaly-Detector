from typing import Dict
import yaml
from pathlib import Path

parent = Path(__file__).parent

def read_yaml(path: str) -> Dict:
    try:
        with open(path) as file:
            return yaml.safe_load(file)
    except yaml.YAMLError as yaml_error:
        print(f'Invalid or corrupted file: {str(yaml_error)}')
        raise yaml_error

    except Exception as e:
        print(f'Error by reading file: {str(e)}')
        raise e

config = read_yaml(parent.joinpath('config.yaml'))
