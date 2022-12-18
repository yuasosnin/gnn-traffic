import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from torch_geometric_temporal.nn.recurrent import A3TGCN
from torch_geometric_temporal.nn.attention import ASTGCN
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR


def get_model(model_name, model_kwargs):
    if model_name == 'A3TGCN':
        return nn.Sequential(
            A3TGCN(in_channels=2, out_channels=model_kwargs['out_channels'], periods=12),
            nn.Linear(model_kwargs['out_channels'], 12))
    elif model_name == 'ASTGCN':
        return ASTGCN(
            nb_block=model_kwargs['nb_block'], 
            in_channels=2, 
            K=3, 
            nb_chev_filter=model_kwargs['nb_chev_filter'], 
            nb_time_filter=model_kwargs['nb_time_filter'], 
            time_strides=1, 
            num_for_predict=12, 
            len_input=12, 
            num_of_vertices=207)


class TemporalGNN(pl.LightningModule):
    def __init__(self, model_name='A3TGCN', lr=1e-3, weight_decay=1e-4, gamma=1, **model_kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = get_model(model_name, model_kwargs)
        self.criterion = nn.MSELoss()

    def forward(self, x, edge_index):
        return self.model(x, edge_index)

    def training_step(self, batch, batch_idx):
        output = self.forward(batch['x'], batch['edge_index'])
        loss = self.criterion(output, batch['y'])
        self.log('train_loss', loss.item(), on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        output = self.forward(batch['x'], batch['edge_index'])
        loss = self.criterion(output, batch['y'])
        self.log('val_loss', loss.item(), on_step=True, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        output = self.forward(batch['x'], batch['edge_index'])
        loss = self.criterion(output, batch['y'])
        self.log('test_loss', loss.item(), on_step=True, on_epoch=True)
        return loss

    def predict_step(self, batch, batch_idx):
        output = self.forward(batch['x'], batch['edge_index'])
        return output

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = ExponentialLR(optimizer, self.hparams.gamma)
        return [optimizer], [scheduler]
