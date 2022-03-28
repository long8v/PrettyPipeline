from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split

from src.dataset.unsmile_dataset import UnsmileDataset


class UnsmileDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/",
        train_val_test_split: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        train_data_path: str = None,
        valid_data_path: str = None,
        pretrained_model: str = None,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        # load datasets only if they're not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            trainset = UnsmileDataset(self.hparams.train_data_path, self.hparams.pretrained_model)
            validset = UnsmileDataset(self.hparams.valid_data_path, self.hparams.pretrained_model)

            dataset = ConcatDataset(datasets=[trainset, validset])
            split_lengths = (
                int(len(dataset) * 0.8),
                int(len(dataset) * 0.1),
                len(dataset) - int(len(dataset) * 0.8) - int(len(dataset) * 0.1),
            )
            self.data_train, self.data_val, self.data_test = random_split(
                dataset=dataset,
                lengths=split_lengths,
                generator=torch.Generator().manual_seed(42),
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
