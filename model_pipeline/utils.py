from pathlib import Path
from typing import List
import pandas as pd

class Utils:

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
