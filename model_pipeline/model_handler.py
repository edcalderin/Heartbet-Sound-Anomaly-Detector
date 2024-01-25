from ts.torch_handler.base_handler import BaseHandler
from zipfile import ZipFile
with ZipFile('model_pipeline.zip') as zip_file:
    zip_file.extractall('.')

from model_pipeline.preprocessing import Preprocessing
import torch
from typing import Tuple
import logging

logging.basicConfig(level = logging.INFO)

class ModelHandler(BaseHandler):

    def preprocess(self, request):
        logging.info('Preprocessing...')
        audio_file = request.get['data']
        melspec = Preprocessing.process_audio(audio_file)
        return melspec.unsqueeze(0)

    def inference(self, x):
        logging.info('Predicting...')
        pred = self.model(x)
        return torch.sigmoid(pred).item()

    def postprocess(self, prediction) -> Tuple:
        logging.info('Postprocessing...')
        threshold: float = 0.5

        return ('abnormal', prediction) \
            if prediction >= threshold \
                else ('normal', 1 - prediction)
