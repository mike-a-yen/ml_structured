from networks.base import BaseNetwork
from datasets.featurized_dataset import FeaturizedDataset
from datasets.trainable_dataset import TrainableDataset

import torch.optim as optim


class BaseModel:
    def __init__(self, featurized_dataset: FeaturizedDataset):
        self.data = TrainableDataset(featurized_dataset)
        self.network = BaseNetwork(self.data.input_shape[1])
