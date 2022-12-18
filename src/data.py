import pytorch_lightning as pl
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader

from torch_geometric_temporal.signal import temporal_signal_split
from torch_geometric_temporal.dataset import METRLADatasetLoader


class SignalDataset(Dataset):
    """
    A simple wrapper for StaticGraphTemporalSignal that allows it to be used with DataLoader.
    """

    def __init__(self, signal):
        super().__init__()
        self._signal = signal

    def __len__(self):
        return self._signal.snapshot_count

    def __getitem__(self, idx):
        return self._signal.__getitem__(idx)


class METRLADataModule(pl.LightningDataModule):
    def __init__(self, raw_data_dir='data', train_steps=12, predict_steps=12, test_size=0.2, batch_size=64, num_workers=0):
        super().__init__()
        self.num_workers = num_workers
        self.batch_size = batch_size

        loader = METRLADatasetLoader(raw_data_dir)
        signal = loader.get_dataset(num_timesteps_in=train_steps, num_timesteps_out=predict_steps)
        train_signal, test_signal = temporal_signal_split(signal, train_ratio=1-test_size*2)
        val_signal, test_signal = temporal_signal_split(test_signal, train_ratio=0.5)

        self.train_datset = SignalDataset(train_signal)
        self.val_datset = SignalDataset(val_signal)
        self.test_datset = SignalDataset(test_signal)

    def train_dataloader(self):
        return DataLoader(self.train_datset, shuffle=True, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.train_datset, shuffle=False, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.train_datset, shuffle=False, batch_size=self.batch_size, num_workers=self.num_workers)
