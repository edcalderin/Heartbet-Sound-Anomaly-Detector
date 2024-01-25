import logging
import os
from pathlib import Path
from zipfile import BadZipFile, ZipFile

from config_management.config_reader import config_params
from config_management.logger import get_logger
from kaggle.api.kaggle_api_extended import KaggleApi

logger = get_logger(
    module_name = 'fetch-dataset', logger_level = logging.INFO, log_location = 'logs')

logger.info('authenticating...')

current_directory = Path(__file__).parent.parent
unzipped_directory = config_params['unzipped_directory']
DATASET_DIRECTORY = current_directory.joinpath(unzipped_directory)

class FetchKaggleDataset:

    def __init__(self) -> None:
        self.api = self.__auth()

    def __auth(self):
        api = KaggleApi()
        api.authenticate()
        return api

    def __exists_dataset(self) -> bool:
        return DATASET_DIRECTORY.exists()

    def __fetch_from_kaggle(self) -> None:
        try:
            kaggle_dataset: str = config_params['kaggle_dataset']

            logger.info(f'downloading dataset "{kaggle_dataset}" from Kaggle')

            self.api.dataset_download_files(kaggle_dataset)

            zip_filename = Path(kaggle_dataset).name
            zip_filename = f'{zip_filename}.zip'

            logger.info('unzipping...')

            with ZipFile(zip_filename) as file:
                file.extractall(path=config_params['unzipped_directory'])

            os.remove(zip_filename)
            logger.info(f'dataset unzipped and saved in "{DATASET_DIRECTORY}/"')

        except BadZipFile:
            logger.error('Not a zip file or a corrupted zip file')
        except Exception as e:
            logger.error(f'Unexpected error {e}')

    def __call__(self):
        if self.__exists_dataset():
            logger.info(
                f'dataset directory with name "{DATASET_DIRECTORY}" already exists')
            return

        self.__fetch_from_kaggle()

if __name__ == '__main__':
    fetcher = FetchKaggleDataset()
    fetcher()
