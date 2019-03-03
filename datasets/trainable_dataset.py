import numpy as np
from datasets.featurized_dataset import FeaturizedDataset

from sklearn.model_selection import train_test_split

class TrainableDataset:
    def __init__(self,featurized_dataset: FeaturizedDataset) -> None:
        self._split_dataset(featurized_dataset)

        self.train_in = np.array(self.train_df.inputs.tolist())
        self.train_out = np.array(self.train_df.target.tolist())
        self.train_prediction = np.zeros_like(self.train_out,dtype=np.float)
        self.train_prediction[:] = np.nan

        self.val_in = np.array(self.val_df.inputs.tolist())
        self.val_out = np.array(self.val_df.target.tolist())
        self.val_prediction = np.zeros_like(self.val_out,dtype=np.float)
        self.val_prediction[:] = np.nan

        self.trainval_in = np.array(self.trainval_df.inputs.tolist())
        self.trainval_out = np.array(self.trainval_df.target.tolist())
        self.trainval_prediction = np.zeros_like(self.trainval_out,dtype=np.float)
        self.trainval_prediction[:] = np.nan

        self.test_in = np.array(self.test_df.inputs.tolist())
        self.test_out = np.array(self.test_df.target.tolist())
        self.test_prediction = np.zeros_like(self.test_out,dtype=np.float)
        self.test_prediction[:] = np.nan

        self.train_size = len(self.train_df)
        self.val_size = len(self.val_df)
        self.test_size = len(self.test_df)

        print(f'Training size: {self.train_size} records.')
        print(f'Validation size: {self.val_size} records.')
        print(f'Test size: {self.test_size} records.')


    def _split_dataset(self, featurized_dataset: FeaturizedDataset) -> None:
        np.random.seed(7)
        self.trainval_df,self.test_df = train_test_split(featurized_dataset.featurized_df,test_size=0.15)
        self.train_df,self.val_df = train_test_split(self.trainval_df,test_size=0.2)

    @property
    def train_shape(self):
        return self.train_in.shape

    @property
    def val_shape(self):
        return self.val_in.shape

    @property
    def test_shape(self):
        return self.test_in.shape
