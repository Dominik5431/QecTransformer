import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from typing import Any, Callable
from torch.optim import Optimizer

from src.data.dataset import DepolarizingSurfaceData, BitflipSurfaceData
from tqdm import tqdm

"""
This script contains the training and inference loop.
"""

def gaussian_kernel(x, y, sigma):
    """Compute the Gaussian kernel between two tensors."""
    if x.dtype is not torch.float32:
        x = x.to(dtype=torch.float32)
    if y.dtype is not torch.float32:
        y = y.to(dtype=torch.float32)
    x_norm = x.pow(2).sum(1).reshape(-1, 1)
    y_norm = y.pow(2).sum(1).reshape(1, -1)
    K = torch.exp(-((x_norm + y_norm - 2.0 * torch.mm(x, y.t())) / (2.0 * sigma ** 2)))
    return K


def mmd_loss(x, y, sigma=1.0):
    """Compute the MMD loss between samples x and y."""
    K_xx = gaussian_kernel(x, x, sigma)
    K_yy = gaussian_kernel(y, y, sigma)
    K_xy = gaussian_kernel(x, y, sigma)
    return K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()


def training_loop(model: nn.Module, dataset, val_set, init_optimizer: Callable[[Any], Optimizer], device, epochs=10,
                  batch_size=100, l=5., mode='depolarizing', activate_scheduler: bool = True,
                  include_mmd: bool = False):
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
    model.to(device)

    optimizer = init_optimizer((model.parameters()))
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=4)

    counter = 0

    for epoch in range(epochs):
        avg_loss = 0
        avg_mmd = 0
        num_batches = 0

        # Training
        model.train()
        for (batch_idx, batch) in enumerate(tqdm(train_loader)):
            # add start token?
            optimizer.zero_grad()

            # Possibility to include MMD loss for training.
            # Slows down training. Sample size needs to be very high to achieve similar mean embedding.
            # Mainly useful if autoregressive transformer is used for both next stabilizer and logical prediction.
            if include_mmd:
                q_samples = model.sample_density()
                mmd = mmd_loss(q_samples, batch)
            else:
                mmd = torch.tensor(0)

            log_prob = model.log_prob(batch)
            loss = torch.mean((-log_prob), dim=0) + l * mmd

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            avg_loss += loss.item()
            avg_mmd += mmd.item()
            num_batches += 1
        avg_loss /= num_batches
        avg_mmd /= num_batches
        print(f"Epoch {epoch + 1}, Loss: {avg_loss}, MMD: {avg_mmd}")

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = 0
            num_batches = 0
            for (batch_idx, batch) in enumerate(val_loader):
                log_prob = model.log_prob(batch)
                # output = model(batch)
                loss = torch.mean((-log_prob), dim=0)
                # loss = criterion(output, batch.float())
                val_loss += loss.item()
                num_batches += 1
            val_loss /= num_batches
            # if val_loss < best_loss:
            #     best_loss = val_loss
            model.save()
            # previous_loss = val_loss
            print(f"Epoch {epoch + 1}, Validation Loss: {val_loss}")
            if activate_scheduler:
                scheduler.step(val_loss)
                print(scheduler.get_last_lr())
                if scheduler.get_last_lr()[0] < 2e-8:
                    counter += 1
                if counter > 10:
                    break
    return model


def eval_log_op(model, distance, noise, device, num=10000, mode='depolarizing'):
    """
    Evaluates the model by predicting logical operators.
    """
    model.eval()
    model.to(device)
    if mode == 'depolarizing':
        data = (DepolarizingSurfaceData(distance=distance,
                                        noise=noise,
                                        name='eval',
                                        load=False,
                                        device=device,
                                        only_syndromes=True)
                .initialize(num))
        # print(data.get_syndromes().size())
        data_1 = data.get_syndromes()
        if not model.readout == 'conv':
            raise NotImplementedError  # TODO: implement here possibility to test convolutional readout vs. decoder readout
        '''
        Only needed, when the logicals are generated autoregressively.
        
        data_2 = torch.cat((data_1, torch.full((data_1.size(0), 1), 0, dtype=torch.long, device=device)), dim=1)
        data_3 = torch.cat((data_1, torch.full((data_1.size(0), 1), 1, dtype=torch.long, device=device)), dim=1)
        pb1 = model.conditioned_forward(data_1)
        pb2_0 = model.conditioned_forward(data_2)
        pb2_1 = model.conditioned_forward(data_3)

        px = pb1 * (1 - pb2_1)
        pz = (1 - pb1) * pb2_0
        py = pb2_1 * pb1
        p1 = (1 - pb1) * (1 - pb2_0)
        '''

        plog = model.predict_logical(data_1)
        py = plog[:, 0] * plog[:, 1]
        px = plog[:, 0] * (1 - plog[:, 1])
        pz = (1 - plog[:, 0]) * plog[:, 1]
        p1 = (1 - plog[:, 0]) * (1 - plog[:, 1])
        # assert (torch.abs(torch.sum(torch.cat((p1, px, pz, py), dim=1), dim=1) - torch.full((px.size(1), 1), 1, device=device)) < 1e-3).all()

        result = np.zeros(num)
        for i in range(num):
            result[i] += px[i] ** 2
            result[i] += py[i] ** 2
            result[i] += pz[i] ** 2
            result[i] += p1[i] ** 2
    else:
        data = (BitflipSurfaceData(distance=distance,
                                   noise=noise,
                                   name='eval',
                                   load=False,
                                   device=device,
                                   only_syndromes=True)
                .initialize(num))
        # print(data.get_syndromes().size())
        data_1 = data.get_syndromes()

        raise NotImplementedError('Bit-flip readout not yet implemented.')

        pb1 = model.predict_logical(data_1)

        px = pb1
        p1 = (1 - pb1)

        result = np.zeros(num)
        for i in range(num):
            result[i] += px[i] ** 2
            result[i] += p1[i] ** 2

    return result
