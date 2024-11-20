import sys

import seaborn as sns
import torch
import numpy as np
from numba_progress import ProgressBar
from tqdm import tqdm

import torch.nn.functional as F
from src.loops import training_loop, eval_log_op
from src.nn.encoder.trade import TraDE
from src.utils import simple_bootstrap, get_pr
from src.optimizer import make_optimizer

from src.data.dataset import DepolarizingSurfaceData, BitflipSurfaceData
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

# t1: try to avoid overfitting and see crossing, compare with analytical expectation
# t2: 1000 epochs
# t3_ bitflip noise
# t4: bitflip on cluster
# t7: back to depolarizing
# t8: significantly simpler model

# tf2: larger log opt, first wo scheduler but with mmd, then opposite, 250 epochs * d
# tf5: adjusted attention
# tf6 adjusted positional encoding

# tf7: full attention on cluster
# tf8: adjacency matrix in attention, positional encoding changed
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
s = 5 #6
s = int(s)

noise = noise_vals[s % 15]
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
num_epochs = 150 * (s // 15 + 1)
batch_size = 1000
data_size = 25000  # val data might not be able to show overfitting
device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')

iteration = 'tf7'
mode = 'depolarizing'
zshot = False

trade_dict = {
    'n': distance ** 2,
    'k': 1,
    'distance': distance,
    'd_model': 32,  # 256,  # 32, 128
    'd_ff': 32,  # 512,  # 32, 256
    'n_layers': 2,
    'n_heads': 8,
    'device': device,
    'dropout': 0.2,
    'vocab_size': 2,
    'max_seq_len': distance ** 2 + 1,
    'noise_model': mode
}

# model = TraDE(name=iteration + '_' + str(distance) + '_' + str(noise), **trade_dict).load().to(device)
'''
# data = torch.tensor([[0, 1, 0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 1, 0, 0], [0, 0, 1, 0, 0, 0, 0, 1]], device=device)
data = (DepolarizingSurfaceData(distance=distance,
                                noise=noise,
                                name=iteration + '_' + str(distance) + '_' + str(noise),
                                load=False,
                                device=device,
                                only_syndromes=True)
        .initialize(5)
        .syndromes)
print(data.size())
print(model.conditioned_forward(data))
exit(-1)
'''


# for name, param in model.named_parameters():
#     print(name, param.size())

def decimal_to_binary_array(decimals, bit_width):
    # Prepare the output array
    n = len(decimals)
    binary_array = np.zeros((n, bit_width), dtype=np.uint8)

    # Loop through each bit position
    for j in range(bit_width):
        # Create a bit mask for the j-th bit
        mask = 1 << (bit_width - j - 1)

        # Extract the j-th bit across all decimal values and store it
        binary_array[:, j] = (decimals & mask) >> (bit_width - j - 1)

    return binary_array


def binary_to_decimal_array(binary):
    # Get the number of binary numbers and their bit width
    binary = binary.ravel().astype(np.uint8)
    decimal_value = 0  # Initialize as an integer
    for j in np.arange(binary.size):
        decimal_value |= binary[j] << (binary.size - j - 1)
    return decimal_value


if task == 0:
    # Create an input example
    '''
    input = (DepolarizingSurfaceData(distance=distance,
                                     noise=noise,
                                     name=iteration + '_' + str(distance) + '_' + str(noise),
                                     load=False,
                                     device=device)
             .initialize(1)
             .syndromes)
    print(input)
    '''
    input = torch.tensor([[1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]],
                         device=device)
    print(input)
    batch_size = 1
    # Prepare a list to store attention weights for each layer
    attention_weights_per_layer = []

    model = TraDE(name=iteration + '_' + str(distance) + '_' + str(noise), **trade_dict).load().to(device)

    # Forward pass through each encoder layer and capture attention weights
    x = input
    x = F.relu(model.fc_in(x))
    x = model.positional_encoding(x)

    seq_len = x.size(1)
    max_len = trade_dict['n'] - 1 + 2 * trade_dict['k']

    mask = torch.zeros(max_len, max_len)
    mask[:max_len - 2 * trade_dict['k'] + 1, :max_len - 2 * trade_dict['k'] + 1] = torch.ones(
        max_len - 2 * trade_dict['k'] + 1, max_len - 2 * trade_dict['k'] + 1)

    for i in range(2 * trade_dict['k']):
        temp = torch.ones(1, max_len)
        temp[0, max_len - 2 * trade_dict['k'] + 1 + i:] = torch.zeros(1, 2 * trade_dict['k'] - 1 - i)
        mask[-2 * trade_dict['k'] + i, :] = temp
    mask = mask.masked_fill(mask == 0, float('-inf'))
    mask = mask.masked_fill(mask == 1, 0.0)
    mask = mask.to(device)
    print(mask)

    for layer in model.encoder.layers:
        # Manually run the forward pass to capture attention weights
        src2, attn_weights = layer.self_attn(x, x, x, attn_mask=mask, average_attn_weights=False)
        attention_weights_per_layer.append(attn_weights)  # Store the attention weights
        src = layer.norm1(x + layer.dropout1(src2))
        src2 = layer.linear2(layer.dropout(layer.activation(layer.linear1(src))))
        src = layer.norm2(src + layer.dropout2(src2))

    # Visualize attention weights for the first layer, first head
    attn_weights = attention_weights_per_layer[0][0, 0].cpu().detach().numpy()  # Select layer 0, batch 0
    src = model.fc_out(src)
    src = F.sigmoid(src).squeeze(2)
    print(src)
    plt.figure(figsize=(10, 8))
    sns.heatmap(attn_weights, cmap="viridis", annot=True, cbar=True)
    plt.title("Attention Weights - Layer 1, Head 1")
    plt.xlabel("Tokens")
    plt.ylabel("Tokens")
    plt.show()
    attn_weights = attention_weights_per_layer[1][0, 2].cpu().detach().numpy()  # Select layer 0, batch 0

    plt.figure(figsize=(10, 8))
    sns.heatmap(attn_weights, cmap="viridis", annot=True, cbar=True)
    plt.title("Attention Weights - Layer 2, Head 3")
    plt.xlabel("Tokens")
    plt.ylabel("Tokens")
    plt.show()
if task == 10:
    if mode == 'depolarizing':
        data_s = torch.empty((data_size, trade_dict['n'] - 1 + 2 * trade_dict['k']), device=device, dtype=torch.long)
        idx = 0
        n_syndromes = 2 ** (distance ** 2 - 1)
        train_syndromes = torch.randperm(n_syndromes)[:30]
        while idx < data_size:
            print('Restart generation.')
            data = (DepolarizingSurfaceData(distance=distance,
                                            noise=noise,
                                            name=iteration + '_' + str(distance) + '_' + str(noise),
                                            load=False,
                                            device=device)
                    .initialize(10 * data_size)
                    .syndromes)
            train_syndromes_s = torch.as_tensor(decimal_to_binary_array(train_syndromes, distance ** 2 - 1),
                                                device=device)
            for s in data:
                if s[:-2] in train_syndromes_s:
                    data_s[idx, :] = s
                    idx += 1
                if idx >= data_size:
                    z_shot_data = (DepolarizingSurfaceData(distance=distance,
                                                           noise=noise,
                                                           name=iteration + '_' + str(distance) + '_' + str(noise) + '_zshot',
                                                           load=False,
                                                           device=device))
                    z_shot_data.syndromes = data_s
                    z_shot_data.save()
                    break
if task == 1:
    if mode == 'bitflip':
        data = (BitflipSurfaceData(distance=distance,
                                   noise=noise,
                                   name=iteration + '_' + str(distance) + '_' + str(noise),
                                   load=False,
                                   device=device)
                .initialize(data_size)
                .save())
    else:
        data = (DepolarizingSurfaceData(distance=distance,
                                        noise=noise,
                                        name=iteration + '_' + str(distance) + '_' + str(noise),
                                        load=False,
                                        device=device)
                .initialize(data_size)
                .save())
elif task == 2:
    if mode == 'bitflip':
        data = (BitflipSurfaceData(distance=distance,
                                   noise=noise,
                                   name=iteration + '_' + str(distance) + '_' + str(noise),
                                   load=True,
                                   device=device)
                .initialize(data_size))
    else:
        data = (DepolarizingSurfaceData(distance=distance,
                                        noise=noise,
                                        name=iteration + '_' + str(distance) + '_' + str(noise) + ('_zshot' if zshot else ''),
                                        load=True,
                                        device=device)
                .initialize(data_size))

    train, val = data.get_train_val_data()  # default ratio 80/20
    try:
        model = TraDE(name=iteration + '_' + str(distance) + '_' + str(noise), **trade_dict).load()  # .load_smaller_d(
        # name=iteration + '_' + str(distance - 2) + '_' + str(noise))
    except FileNotFoundError:
        model = TraDE(name=iteration + '_' + str(distance) + '_' + str(noise), **trade_dict)
    except RuntimeError:
        model = TraDE(name=iteration + '_' + str(distance) + '_' + str(noise), **trade_dict)
    #model = training_loop(model, train, val, make_optimizer(lr), device, epochs=num_epochs, batch_size=batch_size,
    #                     mode=mode, activate_scheduler=False, include_mmd=True)
    #model = training_loop(model, train, val, make_optimizer(lr), device, epochs=2 * num_epochs, batch_size=batch_size,
    #                      mode=mode, activate_scheduler=True)
    # elif task == 20:
    '''
    data = (DepolarizingSurfaceData(distance=distance,
                                    noise=noise,
                                    name=iteration + '_' + str(distance) + '_' + str(noise),
                                    load=True,
                                    device=device)
            .initialize(data_size))
    train, val = data.get_train_val_data()  # default ratio 80/20
    '''
    # model = TraDE(name=iteration + '_' + str(distance) + '_' + str(noise), **trade_dict).load()
    model = training_loop(model, train, val, make_optimizer(lr), device, epochs=2 * num_epochs, batch_size=batch_size,
                          mode=mode, refinement=True)
elif task == 3:
    # Get probabilities for logical operators for each noise value in the form: dict{noise: p(1), p(X), p(Z), p(Y)}
    model = TraDE(name=iteration + '_' + str(distance) + '_' + str(noise), **trade_dict).load()
    res_data = {noise: eval_log_op(model, distance, noise, device, mode=mode)}
    # model = TraDE(name='f2_' + str(distance) + '_' + str(noises[0]), **trade_dict).load()
    # for name, param in model.named_parameters():
    #     print(name, param)
    result = {noise: simple_bootstrap(res_data[noise])}
    torch.save(res_data, "data/data_{0}_{1}_{2}.pt".format(iteration, distance, noise))
    torch.save(result, "data/result_{0}_{1}_{2}.pt".format(iteration, distance, noise))
elif task == 4:
    z = torch.load("data/result_{0}_{1}_{2}.pt".format(iteration, distance, noise))
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

    noises = torch.arange(0, 0.75, 0.01)
    for d in [5]:
        pr = get_pr(d, noises)
        plt.plot(noises, pr, label='d={}'.format(d))
    ax.set_xlabel('noise probability p')
    ax.set_ylabel('participation ratio')

    ax.legend()
    plt.show()
elif task == 100:  # merging dictionaries
    for dist in distances:
        super_dict = {}
        for i, n in enumerate(noise_vals):
            try:
                d = torch.load("data/result_{0}_{1}_{2}.pt".format(iteration, dist, n))
            except FileNotFoundError:
                d = {}
            for k, v in d.items():
                super_dict[k] = v
        torch.save(super_dict, "data/result_{0}_{1}.pt".format(iteration, dist))
elif task == 101:  # plot for different distances
    fig, ax = plt.subplots()
    for i, dist in enumerate(distances):
        dict = torch.load("data/result_{0}_{1}.pt".format(iteration, dist))
        n = list(dict.keys())  # noises
        pr = list(dict.values())  # list of tuples containing mean, uplimit, lowlimit
        # get here median, upper error bar, lower error bar
        pr_m = list(map(lambda x: x[0], pr))
        pr_u = list(map(lambda x: x[1], pr))
        pr_d = list(map(lambda x: x[2], pr))

        # plotting
        ax.errorbar(n, pr_m, yerr=(pr_d, pr_u), marker='o', markersize=4, linestyle='dotted', label='d={}'.format(dist))
    # plt.plot(np.arange(0, 0.4, 0.01), np.arange(0, 0.4, 0.01)**2 + (1 - np.arange(0, 0.4, 0.01))**2, label='physical qubit')
    # plt.xlabel('noise probability p')
    # plt.ylabel('participation ratio')
    # plt.legend()
    # plt.show()
    # elif task == 102:  # find analytical expression for participation ratio
    # first for bit-flip noise
    # fig, ax = plt.subplots()
    noises = np.arange(0.0, 0.4, 0.01)
    for d in [3]:  # , 5]:
        g_stabilizer = np.loadtxt('code/stabilizer_' + 'rsur' + '_d{}_k{}'.format(d, 1))
        logical_opt = np.loadtxt('code/logical_' + 'rsur' + '_d{}_k{}'.format(d, 1))
        pr, entr, var = get_pr(d, noises, g_stabilizer, logical_opt, d ** 2)
        plt.plot(noises, pr, label='d={}'.format(d))
    analytical_5 = np.loadtxt('analytical_d5')
    plt.plot(noises, analytical_5, label='d=5')

    pl = torch.zeros(len(noises))
    for i, n in enumerate(tqdm(noises)):
        sample = (DepolarizingSurfaceData(distance=7,
                                          noise=n,
                                          name=iteration + '_' + str(7) + '_' + str(n),
                                          load=False,
                                          device=device)
                  .initialize(1000)
                  .syndromes).to(torch.float32)
        pl[i] = torch.mean((1 - sample[:, -1]) * (1 - sample[:, -2]))

    plt.plot(noises, pl, label=r'$p(\lambda)_{d=7}$')
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
    noise = 0.4
    data = (DepolarizingSurfaceData(distance=distance,
                                    noise=noise,
                                    name=iteration + '_' + str(distance) + '_' + str(noise),
                                    load=False,
                                    device=device)
            .initialize(10000)
            .syndromes)

    x = 0
    y = 0
    z = 0
    i = 0
    for idx, s in enumerate(tqdm(data)):
        if torch.equal(s.cpu()[-2:], torch.as_tensor([0, 0])):
            i += 1
        if torch.equal(s.cpu()[-2:], torch.as_tensor([1, 0])):
            x += 1
        if torch.equal(s.cpu()[-2:], torch.as_tensor([1, 1])):
            y += 1
        if torch.equal(s.cpu()[-2:], torch.as_tensor([0, 1])):
            z += 1
    print(len(data), i, x, y, z)
