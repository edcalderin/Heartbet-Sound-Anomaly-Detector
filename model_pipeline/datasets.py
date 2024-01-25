from typing import Tuple

import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms

from model_pipeline.preprocessing import Preprocessing


class AudioDataset(Dataset):

    def __init__(
        self, dataframe: pd.DataFrame,
        *,
        audio_length: float,
        target_sample_rate: int) -> None:

        self.__target_sample_rate = target_sample_rate
        self.__num_samples = target_sample_rate * audio_length
        self.__labels = dataframe['label'].values
        self.__filenames = dataframe['fname'].values
        self.__class_indices = {'normal': 0, 'abnormal': 1}

    def __len__(self):
        return len(self.__labels)

    def __getitem__(self, index: int) -> Tuple:
        melspec = Preprocessing.process_audio(self.__filenames[index],
                                              self.__target_sample_rate,
                                              self.__num_samples)
        label: str = self.__labels[index]
        class_idx = self.__class_indices[label]
        return melspec, class_idx

class AudioDatasetWithAugmentation(AudioDataset):

    def __init__(self,
                 dataframe: pd.DataFrame,
                 *,
                 audio_length: float,
                 target_sample_rate: int,
                 transform = transforms.Compose) -> None:

        super().__init__(
            dataframe,
            audio_length = audio_length,
            target_sample_rate = target_sample_rate)

        self.__transform = transform

    def __getitem__(self, index: int) -> dict:
        melspec, class_idx = super().__getitem__(index)
        melspec = self.__transform(melspec)
        return melspec, class_idx
