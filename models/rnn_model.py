from networks.rnn_net import RNNNetwork
from datasets.featurized_dataset import FeaturizedDataset
from datasets.trainable_dataset import TrainableDataset

import torch.optim as optim


class RNNModel:
    def __init__(self, featurized_dataset: FeaturizedDataset):
        self.data = TrainableDataset(featurized_dataset)
        emb_size = 16
        hidden_size = 8
        self.network = RNNNetwork(emb_size,hidden_size)
