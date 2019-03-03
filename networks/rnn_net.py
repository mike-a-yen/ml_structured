import torch
import torch.nn as nn

from vocabulary import character_vocab

class RNNNetwork(nn.Module):
    def __init__(self,emb_size: int = 16, hidden_size: int = 64) -> None:
        nn.Module.__init__(self)
        self.embeds = nn.Embedding(character_vocab.size,emb_size,
                                   padding_idx=character_vocab.PAD_IDX)
        self.rnn = nn.LSTM(emb_size,hidden_size)
        self.dense = nn.Linear(hidden_size,1)
        self.activation = nn.Sigmoid()

    def forward(self,sequences):
        embeddings = self.embeds(sequences) #(N,T,K)
        embeddings = torch.transpose(embeddings,0,1)
        hidden = self._init_hidden(sequences.size(0))
        output,hidden = self.rnn(embeddings,hidden) #(T,N,K), ((N,K), (N,K))
        timesteps,batch_size,K = output.size()
        out_flat = output.view(batch_size*timesteps,K)
        energy = self.dense(out_flat)
        energy = self.activation(energy)
        energy = energy.view(timesteps,batch_size,self.dense.out_features)
        prediction = torch.transpose(energy,0,1)
        return prediction

    def _init_hidden(self,batch_size: int) -> tuple:
        h0 = torch.zeros(self.rnn.num_layers,batch_size,self.rnn.hidden_size).to(self.device)
        c0 = torch.zeros(self.rnn.num_layers,batch_size,self.rnn.hidden_size).to(self.device)
        return h0,c0


    @property
    def device(self):
        p = list(self.rnn.parameters())[0]
        return p.device
