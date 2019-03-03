from pathlib import Path
import pandas as pd
import yaml

from datasets.utils import compute_sha256


_DATASET_METADATA_FILENAME = Path('data/metadata.yaml')

class RawDataset:
    def __init__(self,metadata_filename:Path = _DATASET_METADATA_FILENAME, subsample: int = 0) -> None:
        with open(metadata_filename) as fo:
            metadata = yaml.load(fo)

        self.filename = metadata['filename']
        self.raw_df = pd.read_csv(self.filename)
        self.sha256 = compute_sha256(self.filename)

        if self.sha256!=metadata['sha256']:
            raise Exception('Data inconsistency, hashes do not match!')

        if subsample > 0:
            self._subsample(subsample)

    @property
    def uuid(self) -> str:
        datalength = len(self.raw_df)
        return f'{self.sha256}_{datalength}'

    @property
    def instance_ids(self) -> list:
        return self.raw_df.index.tolist()

    def _subsample(self,subsample):
        self.raw_df = self.raw_df.sample(subsample,random_state=7)
