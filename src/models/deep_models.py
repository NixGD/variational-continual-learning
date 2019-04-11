import torch
import torch.nn
import torch.nn.functional as F


class MultiHeadMLP(torch.nn.Module):
    def __init__(self, x_dim, h_dim, y_dim, n_heads=1, shared_h_dims=(100, 100)):
        super().__init__()
        # check for bad parameters
        if n_heads < 1:
            raise ValueError('Network requires at least one head.')

        self.x_dim = x_dim
        self.h_dim = h_dim
        self.y_dim = y_dim
        self.n_heads = n_heads

        shared_dims = [x_dim] + list(shared_h_dims) + [h_dim]

        # list of layers in shared network
        self.shared_layers = torch.nn.ModuleList([
            torch.nn.Linear(shared_dims[i], shared_dims[i + 1]) for i in
            range(len(shared_dims) - 1)
        ])

        self.heads = torch.nn.ModuleList([
            torch.nn.Linear(self.h_dim, self.y_dim) for _ in range(n_heads)
        ])

        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x, head_idx):
        for layer in self.shared_layers:
            x = F.relu(layer(x))

        x = self.heads[head_idx](x)

        x = self.softmax(x)
        return x

    def prediction(self, x, head_idx):
        return torch.argmax(self(x, head_idx), dim=1)

    def point_estimate_loss(self, x, y, head_idx):
        predictions = self(x, head_idx)
        loss = torch.nn.NLLLoss()(predictions, y)
        return loss

    def vcl_loss(self, x, y, task_length, head_idx):
        self.point_estimate_loss(x, y, head_idx)

    def reset_for_new_task(self):
        pass


class Encoder(torch.nn.Module):
    def __init__(self, x_dim, z_dim, h_dims=(500,)):
        super().__init__()

        self.x_dim = x_dim
        self.z_dim = z_dim
        self.out_dim = 2  # output parameters for a normal

        layer_dims = [x_dim] + list(h_dims) + [z_dim]

        # list of layers in shared network
        self.layers = torch.nn.ModuleList([
            torch.nn.Linear(layer_dims[i], layer_dims[i + 1]) for i in range(len(layer_dims) - 1)
        ])

    def forward(self, x, head_idx):
        z = x
        for layer in self.layers[:-1]:
            z = F.relu(layer(z))
        z = self.layers[-1](z)

        return z

    def encode(self, x, head_idx):
        pass
