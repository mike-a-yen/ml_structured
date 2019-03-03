import torch
import torch.nn as nn

class BaseNetwork(nn.Module):
    def __init__(self, input_size: int) -> None:
        nn.Module.__init__(self)
        self.dense = nn.Linear(input_size,1)
        self.activation = nn.ReLU()

    def forward(self,input):
        energy = self.dense(input)
        prediction = self.activation(energy)
        return prediction
