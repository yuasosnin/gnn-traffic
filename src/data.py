import urllib
from pathlib import Path
from zipfile import ZipFile
import numpy as np

import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, Subset, random_split
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse

# from torch_geometric_temporal.signal import temporal_signal_split
# from torch_geometric_temporal.dataset import METRLADatasetLoader


class METRLADataset(Dataset):
    url = 'https://graphmining.ai/temporal_datasets/METR-LA.zip'

    def __init__(self, root_dir, train_steps=12, predict_steps=12):
        self.root_dir = Path(root_dir)
        self.train_steps = train_steps
        self.predict_steps = predict_steps

        self.download()

        adjacency = torch.tensor(np.load(f'{root_dir}/adj_mat.npy'))
        self.edge_index, self.edge_attr = dense_to_sparse(adjacency)

        self.data = np.load(f'{root_dir}/node_values.npy')
        self.data = torch.tensor(self.data, dtype=torch.float32)
        self.data = self._normalize(self.data, dim=[0,1])

    @staticmethod
    def _normalize(tensor, dim=0):
        std, mean = torch.std_mean(tensor, dim=dim, correction=0)
        return (tensor - mean) / std

    @staticmethod
    def _download_url(url, save_path): 
        with urllib.request.urlopen(url) as dl_file:
            with open(save_path, "wb") as out_file:
                out_file.write(dl_file.read())

    def download(self):
        archive_path = self.root_dir / 'METR-LA.zip'
        if not archive_path.exists(): 
            self.root_dir.mkdir(exist_ok=True)
            self._download_url(self.url, archive_path)

        if not ((self.root_dir/ 'adj_mat.npy').exists() and (self.root_dir/ 'node_values.npy').exists()):
            with ZipFile(archive_path) as zfile:
                zfile.extractall(self.root_dir)

    def __len__(self):
        return len(self.data) - self.train_steps - self.predict_steps + 1

    def __getitem__(self, idx):
        if idx < 0:
            idx = self.__len__() + idx
        x = self.data[idx:idx+self.train_steps]
        y = self.data[idx+self.train_steps:idx+self.train_steps+self.predict_steps]

        x = x.permute(1,2,0)
        y = y[:,:,0].permute(1,0)
        return Data(x=x, edge_index=self.edge_index, edge_attr=self.edge_attr, y=y)


# class SignalDataset(Dataset):
#     """
#     A simple wrapper for StaticGraphTemporalSignal that allows it to be used with DataLoader.
#     """

#     def __init__(self, signal):
#         super().__init__()
#         self._signal = signal

#     def __len__(self):
#         return self._signal.snapshot_count

#     def __getitem__(self, idx):
#         return self._signal.__getitem__(idx)


class METRLADataModule(pl.LightningDataModule):
    def __init__(self, root_dir='data', limit_data=None, train_steps=12, predict_steps=12, test_size=0.2, batch_size=64, num_workers=0):
        super().__init__()
        self.num_workers = num_workers
        self.batch_size = batch_size

        # loader = METRLADatasetLoader(raw_data_dir)
        # signal = loader.get_dataset(num_timesteps_in=train_steps, num_timesteps_out=predict_steps)
        # datset = SignalDataset(signal)

        dataset = METRLADataset(root_dir, train_steps=train_steps, predict_steps=predict_steps)
        if limit_data is not None:
            indices = torch.arange(len(dataset)-limit_data, len(dataset))
            dataset = Subset(dataset, indices)
        lengths = self._get_lengths(len(dataset), test_size)
        self.train_datset, self.val_datset, self.test_datset = random_split(
            dataset, lengths=lengths)

    @staticmethod
    def _get_lengths(length, test_size):
        test_size = round((length*2*test_size) / 2) * 2
        train_size = length - test_size
        val_size = test_size = test_size // 2
        return train_size, val_size, test_size

    def train_dataloader(self):
        return DataLoader(self.train_datset, shuffle=True, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_datset, shuffle=False, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_datset, shuffle=False, batch_size=self.batch_size, num_workers=self.num_workers)
