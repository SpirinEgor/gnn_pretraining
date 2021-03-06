from os import cpu_count
from os.path import join
from typing import Optional

import torch
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from tokenizers import Tokenizer
from torch_geometric.data import DataLoader, Data

from src.data.dataset import GraphDataset


class GraphDataModule(LightningDataModule):
    def __init__(self, data_folder: str, tokenizer: Tokenizer, config: DictConfig):
        super().__init__()
        self.__tokenizer = tokenizer
        self.__config = config
        self.__data_folder = data_folder
        self.__n_workers = cpu_count() if self.__config.num_workers == -1 else self.__config.num_workers

    def train_dataloader(self) -> DataLoader:
        train_dataset_path = join(self.__data_folder, "graphs_train.jsonl.gz")
        train_dataset = GraphDataset(train_dataset_path, self.__tokenizer, self.__config)
        return DataLoader(train_dataset, self.__config.batch_size, num_workers=self.__n_workers, pin_memory=True)

    def val_dataloader(self) -> DataLoader:
        val_dataset_path = join(self.__data_folder, "graphs_val.jsonl.gz")
        val_dataset = GraphDataset(val_dataset_path, self.__tokenizer, self.__config)
        return DataLoader(val_dataset, self.__config.test_batch_size, num_workers=self.__n_workers, pin_memory=True)

    def test_dataloader(self) -> DataLoader:
        test_dataset_path = join(self.__data_folder, "graphs_test.jsonl.gz")
        test_dataset = GraphDataset(test_dataset_path, self.__tokenizer, self.__config)
        return DataLoader(test_dataset, self.__config.test_batch_size, num_workers=self.__n_workers, pin_memory=True)

    def transfer_batch_to_device(self, batch: Data, device: Optional[torch.device] = None) -> Data:
        if device is not None:
            batch = batch.to(device)
        return batch
