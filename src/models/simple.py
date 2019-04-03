import torch
import torch.nn
import torch.nn.functional as F


class MLP(torch.nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.linear_1 = torch.nn.Linear(in_size, 100)
        self.linear_2 = torch.nn.Linear(100, 100)
        self.linear_3 = torch.nn.Linear(100, out_size)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = F.relu(self.linear_1(x))
        x = F.relu(self.linear_2(x))
        x = F.relu(self.linear_3(x))
        x = self.softmax(x)
        return x


class MultiHeadMLP(torch.nn.Module):
    def __init__(self, in_size, out_size, n_heads):
        super().__init__()
        self.shared_layers = torch.nn.ModuleList([
            torch.nn.Linear(in_size, 100),
            torch.nn.Linear(100, 100),
            torch.nn.Linear(100, 100)
        ])

        self.heads = torch.nn.ModuleList([
            torch.nn.Linear(100, out_size) for _ in range(n_heads)
        ])

        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x, head_idx):
        for layer in self.shared_layers:
            x = F.relu(layer(x))

        x = self.heads[head_idx](x)
        # for layer in self.heads[head_idx]:
        #     x = layer(x)

        x = self.softmax(x)
        return x

    def prediction(self, x, head_idx):
        return torch.argmax(self(x, head_idx), dim=1)

    def point_estimate_loss(self, x, y, head_idx):
        predictions = self(x, head_idx)
        loss = torch.nn.CrossEntropyLoss()(predictions, y)
        return loss

    def vcl_loss(self, x, y, task_length, head_idx):
        self.point_estimate_loss(x, y, head_idx)
