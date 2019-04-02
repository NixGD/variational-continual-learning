import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(torch.nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, n_heads, shared_h_dims=(500,), head_h_dims=(500,)):
        super().__init__()

        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.n_heads = n_heads

        shared_dims = [x_dim] + list(shared_h_dims) + [h_dim]
        head_dims = [h_dim] + list(head_h_dims) + [z_dim]

        # list of layers in shared network
        self.shared_layers = nn.ModuleList([
            nn.Linear(shared_dims[i], shared_dims[i + 1]) for i in range(len(shared_dims) - 1)
        ])
        # list of heads, each head is a list of layers
        if n_heads > 0:
            self.heads = nn.ModuleList([
                [nn.Linear(head_dims[i], head_dims[i + 1]) for i in range(len(head_dims) - 1)]
                for _ in range(n_heads)
            ])

    def forward(self, x, head_idx):
        out = x
        for layer in self.shared_layers[:1]:
            out = F.relu(layer(out))

        out = self.shared_layers[-1](out)

        # if network is multi-headed, then last shared layer is internal and needs ReLU
        # also in this case apply head layers
        if self.n_heads > 0:
            out = F.relu(out)

            head = self.heads[head_idx]

            for layer in head[:-1]:
                out = F.relu(layer(out))
            out = head[-1](out)

        return out
