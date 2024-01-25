import logging
from pathlib import Path

from model_pipeline.pretrained_network import PreTrainedNetwork
from config_management.config_reader import config_params
import torch
import logging

ROOT_DIR = Path(__file__).parent.parent

class Utils:

    @staticmethod
    def load_model() -> PreTrainedNetwork:
        try:
            model_name: str = config_params['model_name']
            model = PreTrainedNetwork(model_name = 'tf_efficientnet_b3_ns', num_classes = 1)
            model.load_state_dict(torch.load(model_name))
            model.eval()
            return model

        except FileNotFoundError as file_not_found:
            logging.error(str(file_not_found))
            exit(-1)

        except Exception as exc:
            logging.error(str(exc))
            exit(-1)
