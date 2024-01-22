import logging
from typing import Any, Dict, Tuple

import pandas as pd
import torch
from config_management.logger import get_logger
from pydantic import BaseModel
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import ConcatDataset
from torchaudio.transforms import FrequencyMasking
from torchvision import transforms

from model_training.architecture import PreTrainedNetwork
from model_training.datasets import AudioDataset, AudioDatasetWithAugmentation
from model_training.hearbet_detector_trainer import HearbetDetectorTrainer
from model_training.preprocessing import Preprocessing

logger = get_logger(
    module_name = 'main-module', logger_level = logging.INFO, log_location = 'logs')

logger.info(f'PyTorch version: {torch.__version__}')

SEED: int = 42
AUDIO_LENGTH: int = 10
TARGET_SAMPLE_RATE: int = 4000
AUDIO_DIR: str = 'unzipped_data'
LEARNING_RATE: float = 0.05
EPOCHS: int = 5
BATCH_SIZE: int = 32

params: Dict = {
    'audio_length': AUDIO_LENGTH,
    'target_sample_rate': TARGET_SAMPLE_RATE
}

transformation_classes = transforms.Compose([
    FrequencyMasking(freq_mask_param = 10),
    #TimeStretch(.8, fixed_rate = True),
    #TimeMasking(time_mask_param = 80)
])

class Train(BaseModel):

    pretrained_model: str
    saved_model_dir: str
    data_dir: str

    def __create_datasets(self, dataframe: pd.DataFrame) -> Tuple:
        logger.info('Creating pytorch datasets')

        train_dataset_df, test_dataset_df = train_test_split(
            dataframe,
            train_size = .7,
            random_state = SEED,
            shuffle = True,
            stratify = dataframe.label)

        train_dataset = AudioDataset(dataframe = train_dataset_df, **params)
        test_dataset = AudioDataset(dataframe = test_dataset_df, **params)

        train_dataset_augmented = AudioDatasetWithAugmentation(
            dataframe = train_dataset_df,
            **params,
            transform = transformation_classes)

        test_dataset_augmented = AudioDatasetWithAugmentation(
            dataframe = test_dataset_df,
            **params,
            transform = transformation_classes)

        train_dataset_concatenated = ConcatDataset(
            (train_dataset, train_dataset_augmented))
        test_dataset_concatenated = ConcatDataset(
            (test_dataset, test_dataset_augmented))

        return train_dataset_concatenated, test_dataset_concatenated

    def __call__(self) -> Any:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        dataframe: pd.DataFrame = Preprocessing.create_dataframe(self.data_dir)

        trainset_concatenated, testset_concatenated = self.__create_datasets(dataframe)

        pretrained_model = PreTrainedNetwork(self.pretrained_model, 1)
        pretrained_model = pretrained_model.to(device)

        loss_function = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.SGD(
            params = pretrained_model.parameters(), lr = LEARNING_RATE, momentum = 0.9)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size = 10, gamma = 0.1)

        hearbet_detector_trainer = HearbetDetectorTrainer(
            model = pretrained_model,
            device = device,
            model_dir = self.saved_model_dir)

        hearbet_detector_trainer.fit(
            loss_function = loss_function,
            optimizer = optimizer,
            epochs = EPOCHS,
            batch_size = BATCH_SIZE,
            scheduler = scheduler,
            training_set = trainset_concatenated,
            validation_set = testset_concatenated
        )

if __name__ == '__main__':
    train_model = Train(saved_model_dir = 'pth_models',
                        pretrained_model = 'tf_efficientnet_b3.ns_jft_in1k',
                        data_dir = AUDIO_DIR)
    train_model()
