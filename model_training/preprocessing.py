
from pathlib import Path
from typing import List

import pandas as pd
import torch
import torchaudio
from torch import nn
from torchaudio.transforms import MelSpectrogram, Resample


class Preprocessing:

    @staticmethod
    def process_audio(file_name: str,
                      target_sample_rate: int,
                      num_samples: int) -> torch.Tensor:
        waveform, sample_rate = torchaudio.load(file_name)
        waveform = torch.mean(waveform, axis=0)

        if sample_rate != target_sample_rate:
            resampler = Resample(sample_rate, target_sample_rate)
            waveform = resampler(waveform)

        if waveform.shape[0] >= num_samples:
            waveform = waveform[:num_samples]
        else:
            waveform = nn.functional.pad(waveform, (0, num_samples - waveform.shape[0]))
        melspectrgoram = MelSpectrogram(n_fft = 128, n_mels = 128, hop_length = 128)
        melspec = melspectrgoram(waveform)

        return torch.stack([melspec])

    def __list_files(self, audio_dir: Path):
        for file in audio_dir.glob('**/*.wav'):
            yield file.as_posix()

    @classmethod
    def create_dataframe(cls, audio_dir: str) -> pd.DataFrame:
        data_files = []

        for filename in cls.__list_files(cls, Path(audio_dir)):

            abdnormal_classes: List = ['artifact', 'extrahls', 'extrastole', 'murmur']

            if any(kewword in filename for kewword in abdnormal_classes):
                data_files.append((filename, 'abnormal'))
            elif filename.find('normal')>-1:
                data_files.append((filename, 'normal'))

        return pd.DataFrame(data_files, columns=('fname', 'label'))