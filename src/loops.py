import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Any, Callable
from torch.optim import Optimizer

from src.data.dataset import DepolarizingSurfaceData
from tqdm import tqdm


def training_loop(model: nn.Module, dataset, val_set, init_optimizer: Callable[[Any], Optimizer], device, epochs=10, batch_size=100):
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
    model.to(device)
    model.train()

    optimizer = init_optimizer((model.parameters()))

    best_loss = float("inf")
    previous_loss = float("inf")
    counter = 0

    # include start token, so that first token can attend to something, preferred, see ChatGPT conversation
    start_token_value = 1
    start_token = torch.full((batch_size, 1), start_token_value, dtype=torch.long, device=device)

    for epoch in range(epochs):
        avg_loss = 0
        num_batches = 0
        for (batch_idx, batch) in enumerate(tqdm(train_loader)):
            # add start token
            batch = torch.cat((start_token, batch[:,:-1]), dim=1).to(device)

            optimizer.zero_grad()
            log_prob = model.log_prob(batch)

            loss = torch.mean((-log_prob), dim=0)
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
            num_batches += 1

            if False:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        print(name, param.grad.norm())

                print('Current loss: ', loss.item())

        avg_loss /= num_batches
        print(f"Epoch {epoch + 1}, Loss: {avg_loss}")
        with torch.no_grad():
            val_loss = 0
            num_batches = 0
            for (batch_idx, batch) in enumerate(val_loader):
                batch = torch.cat((start_token, batch[:, :-1]), dim=1).to(device)
                log_prob = model.log_prob(batch)
                loss = torch.mean((-log_prob), dim=0)
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
            if counter > 4:
                break
    return model


def eval_log_op(model, distance, noises, device, num=50000):
    logical = torch.zeros(len(noises), num, 2).to(device)
    model.eval()
    model.to(device)
    for i, noise in enumerate(tqdm(noises)):
        data = (DepolarizingSurfaceData(distance=distance,
                                        noises=[noise],
                                        name='eval',
                                        load=False,
                                        device=device,
                                        only_syndromes=True)
                                        #only_syndromes=False)
                .initialize(num))
        # print(data.get_syndromes().size())

        start_token_value = 1
        start_token = torch.full((num, 1), start_token_value, dtype=torch.long, device=device)

        logical[i] = model.conditioned_forward(torch.cat((start_token, data.get_syndromes()[:, :-1]), dim=1))
    return logical
