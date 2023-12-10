import logging
import os
from pathlib import Path
from zipfile import BadZipFile, ZipFile

from backend_app.config_management.config_reader import config_params
from kaggle.api.kaggle_api_extended import KaggleApi

logging.info('authenticating...')

current_directory = Path(__file__).parent

class FetchKaggleDataset:
    
    def __init__(self) -> None:
        self.api = self.__auth()

    def __auth(self):
        api = KaggleApi()
        api.authenticate()
        return api
    
    def __exists_dataset(self):
        CSV_FILE = current_directory.joinpath(config_params['csv_name'])
        return CSV_FILE.exists()
            
    def __fetch_from_kaggle(self):    
        try:
            kaggle_dataset: str = config_params['kaggle_dataset']
            
            logging.info(f'downloading dataset "{kaggle_dataset}" from Kaggle')
            
            self.api.dataset_download_files(kaggle_dataset)
            
            zip_filename = Path(kaggle_dataset).name
            zip_filename = f'{zip_filename}.zip'
        
            logging.info('unzipping...')
            
            with ZipFile(zip_filename) as file:
                file.extractall()
                csv_file = file.namelist()[0]

            os.remove(zip_filename)
            logging.info(f'dataset unzipped and saved in "{current_directory/csv_file}"')
        
        except BadZipFile:
            logging.error('Not a zip file or a corrupted zip file')
        except Exception as e:
            logging.error(f'Unexpected error {e}')

    def __call__(self):
        if self.__exists_dataset():
            dataset_name: str = current_directory/config_params["csv_name"]
            logging.info(f'dataset with name "{dataset_name}" already exists')
            return
        
        self.__fetch_from_kaggle()
        
if __name__ == '__main__':
    fetcher = FetchKaggleDataset()
    fetcher()