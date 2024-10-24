import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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

        self.fc_in = nn.Embedding(2, self.d_model)
        # self.positional_encoding = PositionalEncoding(self.d_model, self.d_model, self.device, self.dropout)
        # Learnable positional encoding produces better results than RNN based encoding
        self.positional_encoding = LearnablePositionalEncoding(self.n - 1 + 2 * self.k, self.d_model)  # * distance removed
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
        # mask = torch.tril(torch.ones(seq_len, seq_len), diagonal=0)
        # mask = mask.masked_fill(mask == 0, float('-inf'))
        # mask = mask.to(self.device)

        mask = torch.ones(seq_len, seq_len)
        mask = torch.tril(mask, diagonal=0)

        # For subsequent tokens, set diagonal values to 0 to prevent self-attention (except for the start token)
        mask[1:, 1:] = torch.tril(torch.ones((seq_len - 1, seq_len - 1)), diagonal=-1)

        mask = mask.masked_fill(mask == 0, float('-inf'))
        mask = mask.masked_fill(mask == 1, 0.0)
        mask = mask.to(self.device)

        x = self.encoder(x, mask=mask)
        x = self.fc_out(x)
        x = torch.sigmoid(x)
        # scale logits
        # x = x / 10
        return x.squeeze(2)  # return logits

    def log_prob(self, x):
        epsilon = 1e-9

        # Append start token
        start_token_value = 1
        start_token = torch.full((x.size(0), 1), start_token_value, dtype=torch.long, device=self.device)
        x_in = torch.cat((start_token, x[:, :-1]), dim=1).to(self.device)

        x_hat = self.forward(x_in)
        log_prob = torch.log(x_hat + epsilon) * x + torch.log(1 - x_hat + epsilon) * (1 - x)
        return log_prob.sum(dim=1)

    def conditioned_forward(self, syndrome, dtype=torch.int, k=1):
        with torch.no_grad():
            s = syndrome.size(1)
            # x = torch.zeros(syndrome.size(0), s + 2 * self.k, dtype=dtype).to(self.device)
            # x[:, :s] = syndrome
            # x[:, s:] = torch.rand(syndrome.size(0), 2 * self.k).to(self.device)
            logical = torch.zeros(syndrome.size(0), k).to(self.device)  # 2 * * distance removed

            # Append start token
            start_token_value = 1
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


if __name__ == '__main__':
    d = 3
    device = 'mps'
    n = d ** 2
    kwargs_dict = {
        'n': n,
        'k': 1,
        'd_model': 2 * n,
        'd_ff': 2 * n,
        'n_layers': 2,
        'n_heads': 2,
        'device': device,
        'dropout': 0.2,
        'vocab_size': 2,
        'max_seq_len': 50,
    }

    model = TraDE(name='test', **kwargs_dict).to(kwargs_dict['device'])

    device = kwargs_dict['device']

    batch_size = 32
    seq_length = kwargs_dict['n'] + 5
    x = torch.randint(low=0, high=2, size=(batch_size, seq_length)).to(device)  # input
    y = torch.randint(low=0, high=2, size=(batch_size, seq_length)).to(device)  # target

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(5):
        optimizer.zero_grad()
        log_prob = model.log_prob(x)

        loss = torch.mean((-log_prob), dim=0)
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")


    def test(trade):
        res = []
        seq_len = kwargs_dict['n'] + 5
        s0 = torch.ones(1, seq_len, requires_grad=True).to(kwargs_dict['device']).int()
        s = F.relu(trade.fc_in(s0))
        s = trade.positional_encoding(s)
        s.retain_grad()
        for k in range(trade.n):
            mask = torch.tril(torch.ones(seq_len, seq_len), diagonal=0)
            mask = mask.masked_fill(mask == 0, float('-inf'))
            mask = mask.to(device)
            x = trade.encoder(s, mask=mask)
            x = torch.sigmoid(trade.fc_out(x)).squeeze(2)
            loss = x[0, k]
            loss.backward(retain_graph=True)
            grad = s.grad.sum(2)
            print(s.grad)
            depends = (grad[0].cpu().numpy() != 0).astype(np.uint8)
            depends_ix = list(np.where(depends)[0])
            isok = k % trade.n not in depends_ix

            res.append((len(depends_ix), k, depends_ix, isok))

            # pretty print the dependencies
            res.sort()
        for nl, k, ix, isok in res:
            print("output %2d depends on inputs: %70s : %s" % (k, ix, "OK" if isok else "NOTOK"))

    T = TraDE(name='test', **kwargs_dict).to(kwargs_dict['device'])
    test(T)
