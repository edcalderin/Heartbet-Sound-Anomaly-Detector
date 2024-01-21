from typing import Any, Dict, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import ConcatDataset
import torchaudio.transforms as T
from torchvision import transforms
from model_training.architecture import PreTrainedNetwork
from model_training.datasets import AudioDataset, AudioDatasetWithAugmentation
from model_training.hearbet_detector_trainer import HearbetDetectorTrainer
from model_training.preprocessing import Preprocessing
from pydantic import BaseModel
import logging
from config_management.logger import get_logger

logger = get_logger(module_name = 'main-module', logger_level = logging.INFO, log_location = 'logs')

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
    T.FrequencyMasking(freq_mask_param = 10),
    #T.TimeStretch(.8, fixed_rate = True),
    #T.TimeMasking(time_mask_param = 80)
])

class Train(BaseModel):

    pretrained_model_name: str
    saved_model_dir: str
    data_dir: str

    def __create_datasets(self, dataframe: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
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

        train_dataset_concatenated = ConcatDataset((train_dataset, train_dataset_augmented))
        test_dataset_concatenated = ConcatDataset((test_dataset, test_dataset_augmented))

        return train_dataset_concatenated, test_dataset_concatenated

    def __call__(self) -> Any:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        dataframe: pd.DataFrame = Preprocessing.create_dataframe(self.data_dir)

        train_dataset_concatenated, test_dataset_concatenated = self.__create_datasets(dataframe)

        pretrained_model = PreTrainedNetwork(model_name = self.pretrained_model_name, num_classes = 1)
        pretrained_model = pretrained_model.to(device)

        loss_function = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.SGD(params = pretrained_model.parameters(), lr = LEARNING_RATE, momentum = 0.9)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 10, gamma = 0.1)

        hearbetDetectorNetwork = HearbetDetectorTrainer(
            model = pretrained_model,
            device = device,
            model_dir = self.saved_model_dir)

        hearbetDetectorNetwork.fit(
            loss_function = loss_function,
            optimizer = optimizer,
            epochs = EPOCHS,
            batch_size = BATCH_SIZE,
            scheduler = scheduler,
            training_set = train_dataset_concatenated,
            validation_set = test_dataset_concatenated
        )

if __name__ == '__main__':
    train_model = Train(saved_model_dir = 'pth_models',
                        pretrained_model_name = 'tf_efficientnet_b3.ns_jft_in1k',
                        data_dir = AUDIO_DIR)
    train_model()
