import math

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from typing import Any, Callable
from torch.optim import Optimizer

from src.data.dataset import DepolarizingSurfaceData, BitflipSurfaceData
from tqdm import tqdm


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
                  batch_size=100, l=1., mode='depolarizing'):
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
    model.to(device)
    model.train()

    optimizer = init_optimizer((model.parameters()))
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10)

    best_loss = float("inf")
    previous_loss = float("inf")
    counter = 0

    # pos_weight = torch.tensor([2.0]).to(device)
    # criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for epoch in range(epochs):
        avg_loss = 0
        avg_mmd = 0
        num_batches = 0
        for (batch_idx, batch) in enumerate(tqdm(train_loader)):
            # add start token

            optimizer.zero_grad()

            log_prob = model.log_prob(batch)
            q_samples = model.sample_density()
            mmd = mmd_loss(q_samples, batch)
            # output = model(input)
            # print(torch.mean((-log_prob), dim=0), ' vs. ', l * mmd)
            loss = torch.mean((-log_prob), dim=0) + l * mmd
            # loss = criterion(output, batch.float())
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            avg_loss += loss.item()
            avg_mmd += mmd.item()
            num_batches += 1
        avg_loss /= num_batches
        avg_mmd /= num_batches
        print(f"Epoch {epoch + 1}, Loss: {avg_loss}, MMD: {avg_mmd}")

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
            if val_loss > previous_loss:
                counter += 1
            else:
                counter = 0
            previous_loss = val_loss
            print(f"Epoch {epoch + 1}, Validation Loss: {val_loss}")

            scheduler.step(val_loss)
            print(scheduler.get_last_lr())
            # if counter > 4:
            #     break
    return model


def eval_log_op(model, distance, noise, device, num=500, mode='depolarizing'):
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

        data_2 = torch.cat((data_1, torch.full((data_1.size(0), 1), 0, dtype=torch.long, device=device)), dim=1)
        data_3 = torch.cat((data_1, torch.full((data_1.size(0), 1), 1, dtype=torch.long, device=device)), dim=1)
        pb1 = model.conditioned_forward(data_1)
        pb2_0 = model.conditioned_forward(data_2)
        pb2_1 = model.conditioned_forward(data_3)
        # print(data_1)
        # print(pb1)

        px = pb1 * (1 - pb2_1)
        pz = (1 - pb1) * pb2_0
        py = pb2_1 * pb1
        p1 = (1 - pb1) * (1 - pb2_0)

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

        pb1 = model.conditioned_forward(data_1)

        px = pb1
        p1 = (1 - pb1)

        result = np.zeros(num)
        for i in range(num):
            result[i] += px[i] ** 2
            result[i] += p1[i] ** 2

    return result
