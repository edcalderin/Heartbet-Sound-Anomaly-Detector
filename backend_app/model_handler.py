from typing import Dict
import torch
from torch import nn
import torchaudio
from torchaudio.transforms import Resample, MelSpectrogram
from config_management.config_reader import config_params

from model_pipeline import pretrained_network

class ModelHandler:
    def __init__(self, model: pretrained_network, file_content) -> None:
        self._model = model
        self._file_content = file_content
        self._config = config_params
        self._melspectrogram_config = self._config['melspectrogram']

    def _preprocess(self) -> torch.Tensor:
        waveform, sample_rate = torchaudio.load(self._file_content)
        waveform = torch.mean(waveform, axis=0)

        audio_length, target_sample_rate = self._config['audio_length'], self._config['target_sample_rate']
        num_samples: int = target_sample_rate * audio_length

        if sample_rate != target_sample_rate:
            resampler = Resample(sample_rate, target_sample_rate)
            waveform = resampler(waveform)

        if waveform.shape[0] >= num_samples:
            waveform = waveform[:num_samples]
        else:
            waveform = nn.functional.pad(waveform, (0, num_samples - waveform.shape[0]))

        melspectrgoram = MelSpectrogram(n_fft = self._melspectrogram_config['n_fft'],
                                        n_mels = self._melspectrogram_config['n_mels'],
                                        hop_length = self._melspectrogram_config['hop_length'])

        melspec = melspectrgoram(waveform)
        # (1, h, w)
        return torch.stack([melspec])

    def _inference(self, x: torch.Tensor) -> float:
        x = x.unsqueeze(0) # (1, 1, h, w)

        output = self._model(x)

        return torch.sigmoid(output).item()

    def _postprocess(self, x: float) -> Dict:
        threshold: float = self._config['threshold']

        if x >= threshold:
            return {'label': 'abnormal', 'prob': x}
        else:
            return {'label': 'normal', 'prob': 1 - x}

    def __call__(self):
        x = self._preprocess()
        pred: float = self._inference(x)
        return self._postprocess(pred)
