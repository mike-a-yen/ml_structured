from pathlib import Path
import pickle

from datasets.utils import compute_sha256
from datasets.processed_dataset import ProcessedDataset
from vocabulary import encode_character_sequence

_CACHED_DATAPATH = Path('data/featurized')

class FeaturizedDataset:
    def __init__(self,processed_dataset: ProcessedDataset) -> None:
        self.featurized_df = processed_dataset.processed_df
        cached_filename = _CACHED_DATAPATH/f'{processed_dataset.uuid}_featurized.pklb'
        if cached_filename.exists():
            print(f'Loading cached featurized dataset from {cached_filename}')
            with open(cached_filename,'rb') as fo:
                self.__dict__ = pickle.load(fo)
            return

        self.featurize()
        self.uuid = processed_dataset.uuid
        _CACHED_DATAPATH.mkdir(parents=True,exist_ok=True)
        with open(cached_filename, 'wb') as fw:
            pickle.dump(self.__dict__, fw)

    def featurize(self) -> None:
        self.featurized_df['inputs'] = self.featurized_df.name.apply(encode_character_sequence)
        self.featurized_df['target'] = self.featurized_df.inputs.apply(lambda x: [float(chr not in {0,2,3}) for chr in x])
