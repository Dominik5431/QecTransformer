import torch
import numpy as np
from torch.utils.data import Dataset
from abc import ABC, abstractmethod
import logging
from pathlib import Path
from src.error_code.error_code import SurfaceCode


class QECDataset(Dataset, ABC):
    def __init__(self, distance: int, noises, name: str, load: bool, device: torch.device, cluster: bool = False, only_syndromes: bool = False):
        super().__init__()
        self.train = True
        self.device = device
        self.distance = distance
        self.noises = noises
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
        if self.cluster:
            self.syndromes = torch.load("data/syndromes_{0}.pt".format(self.name), mmap=True, map_location=torch.device('cpu'))  # TODO check if mmap reduces memory usage
        else:
            self.syndromes = torch.load("data/syndromes_{0}.pt".format(self.name), mmap=True)
        return self

    def get_syndromes(self):
        return self.syndromes

    @abstractmethod
    def generate_data(self, n, only_syndromes: bool = False):
        raise NotImplementedError


class DepolarizingSurfaceData(QECDataset):
    def __init__(self, distance: int, noises, name: str, load: bool, device: torch.device, cluster: bool = False, only_syndromes: bool = False):
        super().__init__(distance, noises, name, load, device, cluster, only_syndromes)

    def __len__(self):
        return self.syndromes.size(dim=0)

    def __getitem__(self, idx):
        return self.syndromes[idx]

    def generate_data(self, n, only_syndromes: bool = False):
        syndromes = []
        for noise in self.noises:
            c = SurfaceCode(self.distance, noise, 'depolarizing')
            syndromes_noise = c.get_syndromes(n, only_syndromes=only_syndromes)
            syndromes = syndromes + list(syndromes_noise)
        # data is already provided sequentially as [syndromes, noise]
        return torch.as_tensor(np.array(syndromes), device=self.device)

    def get_train_val_data(self, ratio=0.8):
        train_set, val_set = torch.utils.data.random_split(self, [ratio - 1 / len(self), 1 - ratio + 1 / len(self)])
        return train_set, val_set
