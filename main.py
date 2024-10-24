import sys

import torch
import numpy as np
from tqdm import tqdm

from src.loops import training_loop, eval_log_op
from src.nn.encoder.trade import TraDE
from src.utils import make_optimizer, simple_bootstrap, get_pr

from src.data.dataset import DepolarizingSurfaceData
import matplotlib.pyplot as plt

# log
# r4: no log sampling anymore
# r5: changed mask
# r6: changed way of concatenating start token and mask
# r7: mask from qecGPT
# r8: all logical operators included, new loss function
# r9: try everything as in qecGPT to find where error is
# r10: start token appending in forward --> now correct loss function
# r11: new encoding, try to output joint probability of both logical operators

torch.set_printoptions(precision=3, sci_mode=False)

task_dict = {
    'generate data': 1,
    'train': 2,
    'evaluate logical': 3,
    'plot logical': 4
}

# task = task_dict['plot logical']
task = 101
noise_vals = [0.02, 0.05, 0.08, 0.11, 0.14, 0.16, 0.18, 0.20, 0.22, 0.24, 0.27, 0.30, 0.33, 0.36, 0.39]
distances = [3, 5, 7, 9]

# s = sys.argv[1]
s = 9
s = int(s)

noises = [noise_vals[s % 15]]
distance = distances[s // 15]

# noises = [0.11]
# distance = 9

# Hyperparameters
# distance = 3
# noises = np.array(
#     list(map(lambda x: np.exp(-4 / x) / (1 / 3 + np.exp(-4 / x)), np.arange(0.02, 3, 0.02))))
# noises = np.arange(0.2, 0.4, 0.005)
# noises = np.arange(0, 0.41, 0.005)
# noises = [0.1]

lr = 1e-3
num_epochs = 500
batch_size = 100
data_size = 50000
load_data = True
device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')


def get_d_model(dis):
    match dis:
        case 7:
            return 512
        case 9:
            return 512
        case _:
            return 256


trade_dict = {
    'n': distance ** 2,
    'k': 1,
    'distance': distance,
    'd_model': get_d_model(distance),
    'd_ff': get_d_model(distance),
    'n_layers': 2,
    'n_heads': 4,
    'device': device,
    'dropout': 0.2,
    'vocab_size': 2,
    'max_seq_len': 50,
}

if task == 1:
    (DepolarizingSurfaceData(distance=distance,
                             noises=noises,
                             name='f3_' + str(distance) + '_' + str(noises[0]),
                             load=False,
                             device=device)
     .initialize(data_size)
     .save())
    # elif task == 2:
    data = (DepolarizingSurfaceData(distance=distance,
                                    noises=noises,
                                    name='f3_' + str(distance) + '_' + str(noises[0]),
                                    load=True,
                                    device=device)
            .initialize(data_size))
    train, val = data.get_train_val_data()  # default ratio 80/20
    model = TraDE(name='f2_' + str(distance) + '_' + str(noises[0]), **trade_dict).load()
    model = training_loop(model, train, val, make_optimizer(lr), device, epochs=num_epochs, batch_size=batch_size)
# elif task == 3:
    # Get probabilities for logical operators for each noise value in the form: dict{noise: p(1), p(X), p(Z), p(Y)}
    # model = TraDE(name='f3_' + str(distance) + '_' + str(noises[0]), **trade_dict).load()
    res_data = {noises[0]: eval_log_op(model, distance, noises, device)}
    # model = TraDE(name='f2_' + str(distance) + '_' + str(noises[0]), **trade_dict).load()
    # for name, param in model.named_parameters():
    #     print(name, param)
    result = {noises[0]: simple_bootstrap(res_data[noises[0]])}
    torch.save(res_data, "data/data_{0}_{1}_{2}.pt".format('f3', distance, noises[0]))
    torch.save(result, "data/result_{0}_{1}_{2}.pt".format('f3', distance, noises[0]))
elif task == 4:
    z = torch.load("data/result_{0}_{1}_{2}.pt".format('f3', distance, noises[0]))
    n = list(z.keys())  # noises
    pr = list(z.values())  # list of tuples containing mean, uplimit, lowlimit
    # get here median, upper error bar, lower error bar
    pr_m = list(map(lambda x: x[0], pr))
    pr_u = list(map(lambda x: x[1], pr))
    pr_d = list(map(lambda x: x[2], pr))

    # plotting
    fig, ax = plt.subplots()
    ax.errorbar(n, pr_m, yerr=(pr_d, pr_u), marker='o', markersize=4, linestyle='dotted', label='d={}'.format(distance))

    # noises = np.array(list(map(lambda x: 4 / (np.log(3 * (1 - x) / x)), noises)))
    # plt.vlines(1.565, 0, 6, colors='red', linestyles='dashed')
    ax.legend()
    plt.show()
elif task == 100:  # merging dictionaries
    for dist in distances:
        super_dict = {}
        for i, noise in enumerate(noise_vals):
            try:
                d = torch.load("data/result_{0}_{1}_{2}.pt".format('f4', dist, noise))
            except FileNotFoundError:
                d = {}
            for k, v in d.items():
                super_dict[k] = v
        torch.save(super_dict, "data/result_{0}_{1}.pt".format('f4', dist))
elif task == 101:  # plot for different distances
    fig, ax = plt.subplots()
    for i, dist in enumerate(distances):
        dict = torch.load("data/result_{0}_{1}.pt".format('f4', dist))
        n = list(dict.keys())  # noises
        pr = list(dict.values())  # list of tuples containing mean, uplimit, lowlimit
        # get here median, upper error bar, lower error bar
        pr_m = list(map(lambda x: x[0], pr))
        pr_u = list(map(lambda x: x[1], pr))
        pr_d = list(map(lambda x: x[2], pr))

        # plotting
        ax.errorbar(n, pr_m, yerr=(pr_d, pr_u), marker='o', markersize=4, linestyle='dotted', label='d={}'.format(dist))
    # plt.xlabel('noise probability p')
    # plt.ylabel('participation ratio')
    # plt.legend()
    # plt.show()
# elif task == 102:  # find analytical expression for participation ratio
    # first for bit-flip noise
    # fig, ax = plt.subplots()
    noises = torch.arange(0, 0.4, 0.01)
    for d in [3]:
        pr = get_pr(d, noises)
        plt.plot(noises, pr, label='d={}'.format(d))
    ax.set_xlabel('noise probability p')
    ax.set_ylabel('participation ratio')
    ax.legend()
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
    # temps = np.array(list(map(lambda x: 4 / (np.log(3 * (1 - x) / x)), noises)))
    x_log = torch.mean(log, dim=1)[:, 0]
    z_log = torch.mean(log, dim=1)[:, 1]
    plt.plot(noises, x_log.cpu())
    plt.plot(noises, 1 - x_log.cpu())
    plt.show()
elif task == 6:
    model = TraDE(name='r10' + str(distance), **trade_dict).load().to(device)
    s = torch.as_tensor([1, 0, 0, 0, 0, 0, 0, 0]).unsqueeze(0).to(device)
    print(s.size())
    res = model.conditioned_forward(s)
    print(res)
elif task == 7:
    noises = np.array([0.1])
    data = (DepolarizingSurfaceData(distance=distance,
                                    noises=noises,
                                    name='r9' + str(distance),
                                    load=False,
                                    device=device)
            .initialize(10000))
    syndromes = data.get_syndromes()
    zs = 0
    xs = 0
    zs2 = 0
    ss = 0
    no = 0
    for idx, s in enumerate(tqdm(syndromes)):
        if torch.equal(s.cpu()[:8], torch.as_tensor([1, 0, 0, 0, 0, 0, 0, 0])):
            ss += 1
            # print(s)
            if s[-1] == 1:
                zs2 += 1
            if s[-6] == 1:
                xs += 1
            if s[-3] == 1:
                zs += 1
        if torch.equal(s.cpu(), torch.as_tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])):
            no += 1
    print(len(syndromes), ss, xs, zs, zs2, no)
