import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_eunn import EUNN, ModReLU


class Net(nn.Module):
    def __init__(self, name: str, cluster: bool = False):
        super(Net, self).__init__()
        self.name = name
        self.cluster = cluster

    def save(self):
        torch.save(self.state_dict(), "data/net_{0}.pt".format(self.name))
        return self

    def load(self):
        self.load_state_dict(torch.load("data/net_{0}.pt".format(self.name)))
        return self

    def load_smaller_d(self, name: str):
        self.load_state_dict(torch.load("data/net_{0}.pt".format(name)))


class FixedPositionalEncoding(nn.Module):
    def __init__(self, distance: int, d_model, device):
        super().__init__()
        self.distance = distance
        self.device = device

        d_model_half = d_model // 2

        # Create row and column position vector
        height = 2 * distance + 1
        width = height

        row_pos = torch.arange(height, dtype=torch.float32, device=self.device).unsqueeze(1)
        col_pos = torch.arange(width, dtype=torch.float32, device=self.device).unsqueeze(1)

        row_enc = torch.zeros(height, d_model_half, device=self.device)
        col_enc = torch.zeros(width, d_model_half, device=self.device)
        div_term = torch.exp(torch.arange(0, d_model_half, 2, device=self.device) * -(math.log(10000) / d_model_half))

        row_enc[:, 0::2] = torch.sin(div_term * row_pos)
        row_enc[:, 1::2] = torch.cos(div_term * row_pos)
        col_enc[:, 0::2] = torch.sin(div_term * col_pos)
        col_enc[:, 1::2] = torch.cos(div_term * col_pos)

        grid_enc = torch.zeros(height, width, d_model, device=self.device)
        grid_enc[:, :, :d_model_half] = row_enc.unsqueeze(1).expand(-1, width, -1)
        grid_enc[:, :, d_model_half:] = col_enc.unsqueeze(1).expand(-1, width, -1)

        self.pos_enc = torch.zeros(distance ** 2 + 1, d_model,
                                   device=self.device)  # here k when looking into codes that encode several log. qubits

        z_idx = 0
        x_idx = (distance ** 2 - 1) // 2
        for x in range(2, 2 * distance, 4):
            self.pos_enc[z_idx] = grid_enc[x, 0]
            z_idx += 1

        for y in range(2, 2 * distance, 2):
            yi = y % 4
            xs = range(yi, 2 * distance + yi // 2, 2)
            for i, x in enumerate(xs):
                if i % 2 == 0:
                    self.pos_enc[z_idx] = grid_enc[x, y]
                    z_idx += 1
                elif i % 2 == 1:
                    self.pos_enc[x_idx] = grid_enc[x, y]
                    x_idx += 1

        for x in range(4, 2 * distance, 4):
            self.pos_enc[x_idx] = grid_enc[x, 2 * distance]
            x_idx += 1

        enc_log = torch.zeros(d_model, device=self.device)
        for x in range(distance):
            for y in range(distance):
                enc_log += grid_enc[2 * x + 1, 2 * y + 1]
        enc_log = enc_log / (distance ** 2)
        self.pos_enc[-2] = enc_log
        self.pos_enc[-1] = enc_log

    def forward(self, x):
        return x + self.pos_enc[torch.arange(x.size(1), device=x.device)]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, hidden_dim, dev, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(p=dropout)

        self.rnn = nn.GRU(input_size=d_model, hidden_size=hidden_dim, batch_first=True, bidirectional=False, device=dev)

    def forward(self, x):
        x = x + self.dropout(self.rnn(x)[0])
        return x


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, n, d_model):
        # put n here as max_seq_len
        super().__init__()
        self.positional_embedding = nn.Embedding(n, d_model)

    def forward(self, x):
        return x + self.positional_embedding(torch.arange(x.size(1), device=x.device))


class UnitaryEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(UnitaryEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.unitary = EUNN(embedding_dim, num_embeddings)

    def forward(self, x):
        # Convert input indices to one-hot vectors
        one_hot = torch.zeros(
            x.size(0), self.num_embeddings, device=x.device
        ).scatter_(1, x.unsqueeze(1), 1.0)
        #one_hot.pad()
        # Apply custom linear operation
        return ModReLU(self.unitary(one_hot))


class LearnablePositionalEncoding2D(nn.Module):
    def __init__(self, d, d_model, device):  # have also k here, in case for codes with more than 1 logical qubits
        super().__init__()
        self.pos_x = nn.Embedding(2 * d + 1, d_model).to(device)
        self.pos_y = nn.Embedding(2 * d + 1, d_model).to(device)
        self.stab = nn.Embedding(2, d_model).to(device)

        # self.pos_x = UnitaryEmbedding(2 * d + 2, d_model).to(device)
        # self.pos_y = UnitaryEmbedding(2 * d + 2, d_model).to(device)
        # self.stab = UnitaryEmbedding(2, d_model).to(device)

        # self.pos_x.weight.needs_projection = True
        # self.pos_y.weight.needs_projection = True
        # self.stab.weight.needs_projection = True

        self.token_to_x = torch.zeros(d**2 + 1, dtype=torch.long).to(device)
        self.token_to_y = torch.zeros(d**2 + 1, dtype=torch.long).to(device)
        self.token_to_stab = torch.zeros(d**2 + 1, dtype=torch.long).to(device)

        z_idx = 0
        x_idx = (d ** 2 - 1) // 2

        self.token_to_stab[:x_idx] = 0
        self.token_to_stab[x_idx:-2] = 1

        for x in range(2, 2 * d, 4):
            self.token_to_x[x_idx] = x
            self.token_to_y[x_idx] = 0
            x_idx += 1

        for y in range(2, 2 * d, 2):
            yi = y % 4
            xs = range(yi, 2 * d + yi // 2, 2)
            for i, x in enumerate(xs):
                if i % 2 == 0:
                    self.token_to_x[z_idx] = x
                    self.token_to_y[z_idx] = y
                    z_idx += 1
                elif i % 2 == 1:
                    self.token_to_x[x_idx] = x
                    self.token_to_y[x_idx] = y
                    x_idx += 1

        for x in range(4, 2 * d, 4):
            self.token_to_x[x_idx] = x
            self.token_to_y[x_idx] = 2 * d
            x_idx += 1

        self.token_to_x[-2] = d
        self.token_to_y[-2] = d
        self.token_to_stab[-2] = 1  # X denoted as 1, Z as 0!

        self.token_to_x[-1] = d
        self.token_to_y[-1] = d
        self.token_to_stab[-1] = 0

    def forward(self, x):
        return x + (self.pos_x(self.token_to_x[:x.size(1)]) +
                    self.pos_y(self.token_to_y[:x.size(1)]) +
                    self.stab(self.token_to_stab[:x.size(1)]))


class TraDE(Net):
    def __init__(self, name, cluster=False, **kwargs):
        super(TraDE, self).__init__(name, cluster)
        self.n = kwargs['n']
        self.k = kwargs['k']
        self.distance = kwargs['distance']
        self.d_model = kwargs['d_model']
        self.d_ff = kwargs['d_ff']
        self.n_layers = kwargs['n_layers']
        self.n_heads = kwargs['n_heads']
        self.dropout = kwargs['dropout']
        self.device = kwargs['device']
        self.noise_model = kwargs['noise_model']

        self.fc_in = nn.Embedding(3, self.d_model)
        # self.positional_encoding = PositionalEncoding(self.d_model, self.d_model, self.device, self.dropout)

        self.positional_encoding = LearnablePositionalEncoding2D(self.distance, self.d_model, device=self.device)
        # self.l.weight.needs_projection = True

        # Learnable positional encoding produces better results than RNN based encoding
        '''
        if self.noise_model == 'depolarizing':
            self.positional_encoding = LearnablePositionalEncoding(self.n - 1 + 2 * self.k, self.d_model)
        else:
            self.positional_encoding = LearnablePositionalEncoding((self.n - 1) // 2 + self.k, self.d_model)
        
        self.positional_encoding = FixedPositionalEncoding(self.distance, self.d_model, device=self.device)
        '''
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model,
                                                   nhead=self.n_heads,
                                                   dim_feedforward=self.d_ff,
                                                   dropout=self.dropout,
                                                   batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.n_layers)
        self.fc_out = nn.Linear(self.d_model, 1)

    def forward(self, x):
        x = F.relu(self.fc_in(x))
        x = self.positional_encoding(x)

        seq_len = x.size(1)
        max_len = self.n - 1 + 2 * self.k
        '''
        mask = torch.ones(max_len, max_len)
        mask = torch.tril(mask, diagonal=0)

        # For subsequent tokens, set diagonal values to 0 to prevent self-attention (except for the start token)
        mask[1:, 1:] = torch.tril(torch.ones((max_len - 1, max_len - 1)), diagonal=-1)
        '''
        mask = torch.zeros(max_len, max_len)
        mask[:max_len - 2 * self.k + 1, :max_len - 2 * self.k + 1] = torch.ones(max_len - 2 * self.k + 1,
                                                                                max_len - 2 * self.k + 1)
        for i in range(2 * self.k):
            temp = torch.ones(1, max_len)
            temp[0, max_len - 2 * self.k + 1 + i:] = torch.zeros(1, 2 * self.k - 1 - i)
            mask[-2 * self.k + i, :] = temp
        mask = mask.masked_fill(mask == 0, float('-inf'))
        mask = mask.masked_fill(mask == 1, 0.0)

        if seq_len < max_len:
            mask = mask[:seq_len, :seq_len]

        mask = mask.to(self.device)

        x = self.encoder(x, mask=mask)
        x = self.fc_out(x)
        x = torch.sigmoid(x)
        return x.squeeze(2)

    def log_prob(self, x, refinement: bool = False):
        epsilon = 1e-9

        # Append start token
        start_token_value = 2
        start_token = torch.full((x.size(0), 1), start_token_value, dtype=torch.long, device=self.device)
        x_in = torch.cat((start_token, x[:, :-1]), dim=1).to(self.device)

        x_hat = self.forward(x_in)

        if refinement:
            log_prob = torch.log(x_hat[:,-2 * self.k:] + epsilon) * x[:,-2 * self.k:] + torch.log(
                1 - x_hat[:,-2 * self.k:] + epsilon) * (1 - x[:,-2 * self.k:])
            sequence_length = 2
        else:
            log_prob = torch.log(x_hat + epsilon) * x + torch.log(1 - x_hat + epsilon) * (1 - x)
            sequence_length = x.size(1)
        return log_prob.sum(dim=1) / sequence_length

    def conditioned_forward(self, syndrome, dtype=torch.int, k=1):
        with torch.no_grad():
            logical = torch.zeros(syndrome.size(0), k).to(self.device)  # 2 * * distance removed

            # Append start token
            start_token_value = 2
            start_token = torch.full((syndrome.size(0), 1), start_token_value, dtype=torch.long, device=self.device)
            syndrome = torch.cat((start_token, syndrome), dim=1).to(self.device)

            for i in range(k):  # 2 * * distance removed
                conditional = self.forward(syndrome)
                # conditional = torch.sigmoid(logits)
                if len(conditional.shape) < 2:
                    conditional = conditional.unsqueeze(0)
                # r = torch.as_tensor(np.random.rand(syndrome.size(0)), dtype=torch.float32, device=self.device)
                # syndrome = torch.cat((syndrome, 1*(r < logical[:, i]).unsqueeze(1)), dim=1)
                logical[:, i] = conditional[:, -1]
                # x[:, s + i] = torch.floor(2 * conditional[:, s + i])
                # x[:, s + i] = conditional[:, s + i]
        return logical

    def sample_density(self, num_samples=1000):
        if self.noise_model == 'bitflip':
            sequence_length = (self.n - 1) // 2 + self.k
        else:
            sequence_length = (self.n - 1) + 2 * self.k
        samples = torch.zeros(num_samples, sequence_length, dtype=torch.long).to(self.device)
        with torch.no_grad():
            for i in torch.arange(sequence_length):
                probs = self.forward(samples)
                p_i = probs[:, i]

                next_token = torch.bernoulli(p_i)

                samples[:, i] = next_token

        return samples


if __name__ == '__main__':
    c = FixedPositionalEncoding(3, 128)
