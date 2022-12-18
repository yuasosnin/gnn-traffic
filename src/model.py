import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from torch_geometric_temporal.nn.recurrent import A3TGCN
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR


class TemporalGNN(pl.LightningModule):
    def __init__(self, in_features, hidden_features=32, periods=12, lr=1e-3, weight_decay=1e-4, gamma=1):
        super().__init__()
        self.save_hyperparameters()

        self.tgnn = A3TGCN(
            in_channels=in_features, 
            out_channels=hidden_features, 
            periods=periods)
        self.linear = nn.Linear(hidden_features, periods)
        self.act = nn.ReLU()
        self.criterion = nn.MSELoss()

    def forward(self, x, edge_index):
        h = self.tgnn(x, edge_index)
        h = self.act(h)
        h = self.linear(h)
        return h

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
