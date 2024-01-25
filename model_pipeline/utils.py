from pathlib import Path
from typing import List
import pandas as pd

class Utils:

    @staticmethod
    def create_dataframe(audio_dir: str) -> pd.DataFrame:
        data_files = []

        for filename in Path(audio_dir).glob('**/*.wav'):

            filename_ = filename.as_posix()

            abdnormal_classes: List = ['artifact', 'extrahls', 'extrastole', 'murmur']

            if any(kewword in filename_ for kewword in abdnormal_classes):
                data_files.append((filename_, 'abnormal'))
            elif filename_.find('normal')>-1:
                data_files.append((filename_, 'normal'))

        return pd.DataFrame(data_files, columns=('fname', 'label'))
