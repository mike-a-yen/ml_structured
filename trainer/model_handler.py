import numpy as np
from typing import Union

import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda as cuda

from tqdm import tqdm
import wandb

from models.base_model import BaseModel
from models.rnn_model import RNNModel

def _get_torch_device() -> torch.device:
    if cuda.is_available():
        ngpus = cuda.device_count()
        print(f'CUDA is available with {ngpus} devices.')
        return torch.device('cuda:0')
    return torch.device('cpu')

class ModelHandler:
    def __init__(self, model: RNNModel,use_wandb: bool = True):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self.use_wandb = use_wandb

        self.device = _get_torch_device()
        self.model = model
        self.model.network.to(self.device)
        self.opt = optim.Adam(self.model.network.parameters(),lr=0.001)
        self.loss_fn = nn.BCELoss(reduction='sum')
        self.batch_size = 128

        self.train_indices = np.arange(self.model.data.train_size)

        if self.use_wandb:
            wandb.init("torch_ocr")
            wandb.watch(self.model.network)

    def fit(self,max_epochs: int = 8) -> None:
        for epoch in range(max_epochs):
            print('Epoch {}/{}:'.format(epoch+1,max_epochs),end=' ')
            train_cost = self._fit_epoch()
            val_cost,val_acc = self._eval_epoch()
            log_params = {
                    'epoch':epoch,
                    'train_loss':train_cost.item(),
                    'val_loss':val_cost.item(),
                    'val_acc':val_acc.item()
            }
            print('Training Loss: {:0.5f} Validation Loss: {:0.5f} Validation Accuracy: {:0.4f}'.format(train_cost.item(),val_cost.item(),val_acc.item()))
            if self.use_wandb:
                wandb.log(log_params)
            #self.inspect()
        return

    def _fit_batch(self,X: torch.LongTensor,y: torch.FloatTensor) -> torch.FloatTensor:
        cost = self._eval_batch(X,y)
        cost.backward()
        self.opt.step()
        return cost

    def _eval_batch(self,X: torch.LongTensor,y: torch.FloatTensor) -> torch.FloatTensor:
        predictions = self.model.network(X)
        return self._get_cost(predictions,y)

    def _get_cost(self, predictions: torch.FloatTensor, y: torch.FloatTensor) -> torch.FloatTensor:
        flat_pred = predictions.contiguous().view(-1,1)
        flat_y = y.contiguous().view(-1,1)
        cost = self.loss_fn(flat_pred,flat_y)
        return cost

    def _get_accuracy(self, predictions: torch.FloatTensor, y: torch.FloatTensor) -> torch.FloatTensor:
        threshold = 0.5
        flat_pred = predictions.contiguous().view(-1,1)
        flat_y = y.contiguous().view(-1,1)
        guess = (flat_pred>threshold).float()
        is_eq = guess.eq(flat_y).float()
        return is_eq.mean()

    def _fit_epoch(self) -> torch.FloatTensor:
        np.random.shuffle(self.train_indices)
        self.model.network.train()
        epoch_cost = 0.0
        n_batches = self.n_batches(self.model.data.train_shape[0])
        with tqdm(total=n_batches,ncols=80,desc='Training ',leave=False,unit='batch') as pbar:
            for batch_id in range(n_batches):
                idxs = self._get_training_batch_indices(batch_id)
                X = torch.LongTensor(self.model.data.train_in[idxs]).to(self.device)
                y = torch.FloatTensor(self.model.data.train_out[idxs]).to(self.device)
                predictions = self.model.network(X)
                self.model.data.train_prediction[idxs] = predictions.cpu().detach().numpy().squeeze()
                batch_cost = self._get_cost(predictions,y)
                batch_cost.backward()
                self.opt.step()
                epoch_cost += batch_cost
                pbar.update(1)
        return epoch_cost/self.model.data.train_size

    def _eval_epoch(self):
        self.model.network.eval()
        epoch_cost = 0.0
        epoch_accuracy = 0.0
        n_batches = self.n_batches(self.model.data.val_shape[0])
        with tqdm(total=n_batches,ncols=80,desc='Validating ',leave=False,unit='batch') as pbar:
            for batch_id in range(n_batches):
                offset = self.batch_size*batch_id
                idxs = np.arange(offset,offset+self.batch_size)
                X = torch.LongTensor(self.model.data.train_in[idxs]).to(self.device)
                y = torch.FloatTensor(self.model.data.train_out[idxs]).to(self.device)
                predictions = self.model.network(X)
                self.model.data.val_prediction[idxs] = predictions.cpu().detach().numpy().squeeze()
                batch_cost = self._get_cost(predictions,y)
                epoch_cost += batch_cost
                batch_accuracy = self._get_accuracy(predictions,y)
                epoch_accuracy += batch_accuracy
                pbar.update(1)
        self.opt.zero_grad()
        return epoch_cost/self.model.data.val_size, epoch_accuracy/n_batches

    def _get_training_batch_indices(self,batch_id):
        offset = batch_id*self.batch_size
        return self.train_indices[offset:offset+self.batch_size]

    def n_batches(self,n_records):
        main = n_records//self.batch_size
        remainder = int(n_records/self.batch_size-main == 0)
        return main+remainder

    def inspect(self):
        idxs = np.where(self.model.data.train_prediction!=self.model.data.train_out)[0]
        idx = np.random.choice(idxs)
        record = self.model.data.train_df.iloc[idx]
        name = record['name']
        target = record['target']
        prediction = self.model.data.train_prediction[idx]
        print({'Name':name,'target':target,'prediction':prediction})
