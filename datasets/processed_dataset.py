from pathlib import Path
from PIL import Image

import numpy as np
import pickle

from datasets.utils import compute_sha256
from datasets.raw_dataset import RawDataset

_TRAIN_DATA_DIR = 'data/train'
_VAL_DATA_DIR = 'data/val'
_TEST_DATA_DIR = 'data/test'

_CACHED_DATAPATH = Path('data/processed/')

class ProcessedDataset:
    def __init__(self,raw_dataset: RawDataset) -> None:
        cached_filename = _CACHED_DATAPATH / f'{raw_dataset.uuid}_processed.pklb'
        if cached_filename.exists():
            print(f'Loading processed dataset from {cached_filename}')
            with open(cached_filename,'rb') as fo:
                self.__dict__ = pickle.load(fo)
                return

        self.raw_dataset = raw_dataset
        self.clean_and_transform()

        self.params = _get_dataset_params(self)
        self.uuid = raw_dataset.uuid
        _CACHED_DATAPATH.mkdir(parents=True, exist_ok=True)
        with open(cached_filename, 'wb') as fw:
            pickle.dump(self.__dict__, fw)


    def clean_and_transform(self) -> None:
        self._clean()
        self._transform()

    def _transform(self) -> None:
        self.processed_df = self.raw_dataset.raw_df.copy()
        self.processed_df['type'] = self.processed_df.FILENAME.apply(lambda x: x.split('_')[0].lower())
        self.processed_df['image_path'] = 'data/'+self.processed_df.type+'/'+self.processed_df.FILENAME
        self.processed_df['name_length'] = self.processed_df.IDENTITY.apply(len)
        self.processed_df['image_size'] = self.processed_df.image_path.apply(_get_image_size)
        self.processed_df['name'] = self.processed_df.IDENTITY.str.lower()


    def _clean(self) -> None:
        self.raw_dataset.raw_df.dropna(inplace=True)


def _get_dataset_params(dataset: ProcessedDataset) -> dict:
    params = dict(
        n_records = len(dataset.processed_df),
        max_name_length = max(dataset.processed_df.name_length),
        min_name_length = min(dataset.processed_df.name_length),
        type = list(dataset.processed_df.type.unique()),
    )
    return params

def _get_image_size(filename: str) -> tuple:
    im = np.array(Image.open(filename))
    return im.shape
