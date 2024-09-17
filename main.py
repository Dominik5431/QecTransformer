import torch
import numpy as np
from tqdm import tqdm

from src.loops import training_loop, eval_log_op
from src.nn.encoder.trade import TraDE
from src.utils import make_optimizer

from src.data.dataset import DepolarizingSurfaceData
import matplotlib.pyplot as plt

# log
# r4: no log sampling anymore
# r5: changed mask
# r6: changed way of concatenating start token and mask
# r7: mask from qecGPT

task_dict = {
    'generate data': 1,
    'train': 2,
    'evaluate logical': 3,
    'plot logical': 4
}

task = task_dict['evaluate logical']
task = 7

# Hyperparameters
distance = 3
# noises = np.array(
#     list(map(lambda x: np.exp(-4 / x) / (1 / 3 + np.exp(-4 / x)), np.arange(0.02, 3, 0.02))))
noises = np.arange(0.2, 0.4, 0.005)
# noises = np.arange(0, 0.41, 0.01)

lr = 0.001
num_epochs = 20
batch_size = 100
data_size = 5000
load_data = True
device = torch.device('mps')

trade_dict = {
    'n': distance ** 2,
    'k': 1,
    'd_model': 32,
    'd_ff': 64,
    'n_layers': 2,
    'n_heads': 4,
    'device': device,
    'dropout': 0.,
    'vocab_size': 2,
    'max_seq_len': 50,
}

if task == 1:
    (DepolarizingSurfaceData(distance=distance,
                             noises=noises,
                             name='r4_wo_log' + str(distance),
                             load=False,
                             device=device)
     .initialize(data_size)
     .save())
elif task == 2:
    data = (DepolarizingSurfaceData(distance=distance,
                                    noises=noises,
                                    name='r4_wo_log' + str(distance),
                                    load=True,
                                    device=device)
            .initialize(data_size))
    train, val = data.get_train_val_data()  # default ratio 80/20
    model = TraDE(name='r7' + str(distance), **trade_dict)  # .load()
    model = training_loop(model, train, val, make_optimizer(lr), device)
elif task == 3:
    model = TraDE(name='r7' + str(distance), **trade_dict).load()
    for name, param in model.named_parameters():
        print(name, param)
    result = eval_log_op(model, distance, noises, device)
    torch.save(result, "data/results_{0}_{1}.pt".format('r7', distance))
elif task == 4:
    log = torch.load("data/results_{0}_{1}.pt".format('r7', distance))
    x_log = torch.mean(log, dim=1)[:, 0]
    z_log = torch.mean(log, dim=1)[:, 1]
    plt.plot(noises, x_log.cpu())
    plt.plot(noises, 1 - x_log.cpu())
    plt.show()
elif task == 5:
    log = torch.load("data/results_{0}_{1}.pt".format('r5', 3))
    noises = np.arange(0., 0.41, 0.01)
    x_log = torch.mean(log, dim=1)[:, 0]
    z_log = torch.mean(log, dim=1)[:, 1]
    plt.plot(noises, x_log.cpu())
    plt.plot(noises, 1 - x_log.cpu())
    log = torch.load("data/results_{0}_{1}.pt".format('r5', 5))
    noises = np.arange(0., 0.4, 0.005)
    x_log = torch.mean(log, dim=1)[:, 0]
    z_log = torch.mean(log, dim=1)[:, 1]
    plt.plot(noises, x_log.cpu())
    plt.plot(noises, 1 - x_log.cpu())
    plt.show()
elif task == 6:
    model = TraDE(name='r7' + str(distance), **trade_dict).load().to(device)
    s = torch.as_tensor([1, 1, 1, 1, 1, 1, 1, 1]).unsqueeze(0).to(device)
    print(s.size())
    res = model.conditioned_forward(s)
    print(res)
elif task == 7:
    noises = np.array([0.01])
    data = (DepolarizingSurfaceData(distance=distance,
                                    noises=noises,
                                    name='r4_wo_log' + str(distance),
                                    load=False,
                                    device=device)
            .initialize(10000))
    syndromes = data.get_syndromes()
    zs = 0
    xs = 0
    xs2 = 0
    ss = 0
    no = 0
    for idx, s in enumerate(tqdm(syndromes)):
        if torch.equal(s.cpu()[:-3], torch.as_tensor([1, 0, 0, 0, 0, 0, 0, 0])):
            ss += 1
            print(s)
            if s[-1] == 1:
                xs2 += 1
            if s[-2] == 1:
                xs += 1
            if s[-3] == 1:
                zs += 1
        if torch.equal(s.cpu(), torch.as_tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])):
            no += 1
    print(len(syndromes), ss, xs, xs2, zs, no)


