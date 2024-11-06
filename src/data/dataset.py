import torch
import numpy as np
from torch.utils.data import Dataset
from abc import ABC, abstractmethod
import logging
from pathlib import Path
from src.error_code.error_code import SurfaceCode


class QECDataset(Dataset, ABC):
    def __init__(self, distance: int, noise, name: str, load: bool, device: torch.device, cluster: bool = False, only_syndromes: bool = False):
        super().__init__()
        self.train = True
        self.device = device
        self.distance = distance
        self.noise = noise
        self.name = name
        self.syndromes = None
        self.load_data = load
        self.cluster = cluster
        self.only_syndromes = only_syndromes
        if not (type(name) is str):
            raise ValueError

    def initialize(self, num: int):
        if self.load_data:
            try:
                self.load()
            except NameError:
                logging.error("No valid noise model specified.")
        else:
            self.syndromes = self.generate_data(num, self.only_syndromes)
        return self

    def training(self):
        self.train = True
        return self

    def eval(self):
        self.train = False
        return self

    def save(self):
        torch.save(self.syndromes, "data/syndromes_{0}.pt".format(self.name))
        return self

    def load(self):
        self.syndromes = torch.load("data/syndromes_{0}.pt".format(self.name), map_location=torch.device(self.device))
        return self

    def get_syndromes(self):
        return self.syndromes

    @abstractmethod
    def generate_data(self, n, only_syndromes: bool = False):
        raise NotImplementedError


class DepolarizingSurfaceData(QECDataset):
    def __init__(self, distance: int, noise, name: str, load: bool, device: torch.device, cluster: bool = False, only_syndromes: bool = False):
        super().__init__(distance, noise, name, load, device, cluster, only_syndromes)

    def __len__(self):
        return self.syndromes.size(dim=0)

    def __getitem__(self, idx):
        return self.syndromes[idx]

    def generate_data(self, n, only_syndromes: bool = False):
        syndromes = []
        c = SurfaceCode(self.distance, self.noise, noise_model='depolarizing')
        syndromes = c.get_syndromes(n, only_syndromes=only_syndromes)
        # data is already provided sequentially as [syndromes, noise]
        return torch.as_tensor(np.array(syndromes), device=self.device)

    def get_train_val_data(self, ratio=0.8):
        train_set, val_set = torch.utils.data.random_split(self, [ratio - 1 / len(self), 1 - ratio + 1 / len(self)])
        return train_set, val_set


class BitflipSurfaceData(QECDataset):
    def __init__(self, distance: int, noise, name: str, load: bool, device: torch.device, cluster: bool = False, only_syndromes: bool = False):
        super().__init__(distance, noise, name, load, device, cluster, only_syndromes)

    def __len__(self):
        return self.syndromes.size(dim=0)

    def __getitem__(self, idx):
        return self.syndromes[idx]

    def generate_data(self, n, only_syndromes: bool = False):
        syndromes = []
        c = SurfaceCode(self.distance, self.noise, noise_model='bitflip')
        syndromes = c.get_syndromes(n, only_syndromes=only_syndromes)
        syndromes = torch.as_tensor(syndromes, device=self.device)
        # print(syndromes[0:4])
        # data is already provided sequentially as [syndromes, noise]
        if only_syndromes:
            syndromes_bf = syndromes[:, :int((self.distance**2 - 1) / 2)]
        else:
            syndromes_bf = torch.cat((syndromes[:, :int((self.distance**2 - 1) / 2)], syndromes[:, -1].unsqueeze(1)), dim=1)
        return syndromes_bf

    def get_train_val_data(self, ratio=0.8):
        train_set, val_set = torch.utils.data.random_split(self, [ratio - 1 / len(self), 1 - ratio + 1 / len(self)])
        return train_set, val_set
