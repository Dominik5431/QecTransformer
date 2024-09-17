import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, hidden_dim, dev, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(p=dropout)

        self.rnn = nn.GRU(input_size=d_model, hidden_size=hidden_dim, batch_first=True, bidirectional=False, device=dev)

    def forward(self, x):
        x = x + self.dropout(self.rnn(x)[0])
        return x


class TraDE(nn.Module):
    def __init__(self, *args, **kwargs):
        super(TraDE, self).__init__()
        self.n = kwargs['n']
        self.d_model = kwargs['d_model']
        self.d_ff = kwargs['d_ff']
        self.n_layers = kwargs['n_layers']
        self.n_heads = kwargs['n_heads']
        self.dropout = kwargs['dropout']
        self.device = kwargs['device']

        self.fc_in = nn.Embedding(2, self.d_model)
        self.positional_encoding = PositionalEncoding(self.d_model, self.d_model, self.device, self.dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model,
                                                   nhead=self.n_heads,
                                                   dim_feedforward=self.d_ff,
                                                   dropout=self.dropout,
                                                   batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.n_layers)
        self.fc_out = nn.Linear(self.d_model, 1)

        self.register_buffer('mask', torch.ones(self.n, self.n))
        self.mask = torch.tril(self.mask, diagonal=0)
        self.mask = self.mask.masked_fill(self.mask == 0, float('-inf'))

    def forward(self, x):
        x = F.relu(self.fc_in(x))
        x = self.positional_encoding(x)
        x = self.encoder(x, mask=self.mask)
        x = torch.sigmoid(self.fc_out(x))
        return torch.squeeze(x)

    def log_prob(self, x):
        x_hat = self.forward(x)
        log_prob = torch.log(x_hat) * x + torch.log(1 - x_hat) * (1 - x)
        return log_prob.sum(dim=1)


if __name__ == '__main__':
    d = 3
    device = 'mps'
    n = d ** 2
    kwargs_dict = {
        'n': n,
        'd_model': 2 * n,
        'd_ff': 2 * n,
        'n_layers': 2,
        'n_heads': 2,
        'device': device,
        'dropout': 0.2,
        'vocab_size': 2,
        'max_seq_len': 50,
    }

    model = TraDE(**kwargs_dict).to(kwargs_dict['device'])

    device = kwargs_dict['device']

    batch_size = 32
    seq_length = kwargs_dict['n']
    x = torch.randint(low=0, high=2, size=(batch_size, seq_length)).to(device)  # input
    y = torch.randint(low=0, high=2, size=(batch_size, seq_length)).to(device)  # target

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(10):
        optimizer.zero_grad()
        log_prob = model.log_prob(x)

        loss = torch.mean((-log_prob), dim=0)
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

    def test(trade):
        res = []
        s0 = torch.ones(1, kwargs_dict['n'], requires_grad=True).to(kwargs_dict['device']).int()
        s = F.relu(trade.fc_in(s0))
        s = trade.positional_encoding(s)
        s.retain_grad()
        for k in range(trade.n):
            x = trade.encoder(s, mask=trade.mask)
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


    # T = TraDE(**kwargs_dict).to(kwargs_dict['device'])
    # test(T)



