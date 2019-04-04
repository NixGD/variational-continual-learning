from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
import util.operations
from layers.variational import VariationalLayer, MeanFieldGaussianLinear
import math

EPSILON = 1e-20  # Small value to avoid divide-by-zero and log(zero) problems


class VCL(nn.Module, ABC):
    def __init__(self, epsilon=EPSILON):
        super().__init__()
        self.epsilon = epsilon

    def prediction(self, x, head):
        return torch.argmax(self.forward(x, head), dim=1)

    @abstractmethod
    def reset_for_new_task(self, head_idx):
        pass

    @abstractmethod
    def vcl_loss(self, x, y, head_idx, task_size):
        pass

    @abstractmethod
    def point_estimate_loss(self, x, y, head_idx):
        pass

    @abstractmethod
    def get_statistics(self) -> (list, dict):
        pass

    @abstractmethod
    def _kl_divergence(self, head_idx) -> float:
        pass


class DiscriminativeVCL(VCL):
    """
    A Bayesian neural network which is updated using variational inference
    methods, and which is multi-headed at the output end. Suitable for
    continual learning of discriminative tasks.
    """

    def __init__(self, x_dim, h_dim, y_dim, n_heads=1, shared_h_dims=(100, 100),
                 initial_posterior_variance=1e-6, mc_sampling_n=10, device='cpu'):
        super().__init__()
        # check for bad parameters
        if n_heads < 1:
            raise ValueError('Network requires at least one head.')

        self.x_dim = x_dim
        self.h_dim = h_dim
        self.y_dim = y_dim
        self.n_heads = n_heads
        self.ipv = initial_posterior_variance
        self.mc_sampling_n = mc_sampling_n
        self.device = device

        shared_dims = [x_dim] + list(shared_h_dims) + [h_dim]

        # list of layers in shared network
        self.shared_layers = nn.ModuleList([
            MeanFieldGaussianLinear(shared_dims[i], shared_dims[i + 1], self.ipv, EPSILON) for i in range(len(shared_dims) - 1)
        ])
        # list of heads, each head is a list of layers
        self.heads = nn.ModuleList([
            MeanFieldGaussianLinear(self.h_dim, self.y_dim, self.ipv, EPSILON) for _ in range(n_heads)
        ])

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, head_idx=0, sample_parameters=True):
        x_out = torch.zeros(size=(x.size()[0], self.y_dim)).to(self.device)

        # repeat forward pass n times to sample layer params multiple times
        for _ in range(self.mc_sampling_n if sample_parameters else 1):
            x_in = x
            # shared part
            for layer in self.shared_layers:
                x_in = F.relu(layer(x_in, sample_parameters=sample_parameters))

            # head
            x_in = self.heads[head_idx](x_in, sample_parameters=sample_parameters)
            x_in = self.softmax(x_in)

            x_out.add_(x_in)

        x_out.div_(self.mc_sampling_n)

        return x_out

    def vcl_loss(self, x, y, head_idx, task_size):
        return self._kl_divergence(head_idx) / task_size + torch.nn.NLLLoss()(self(x, head_idx), y)

    def point_estimate_loss(self, x, y, head_idx=0):
        return torch.nn.NLLLoss()(self(x, head_idx, sample_parameters=False), y)

    def prediction(self, x, task=0):
        """ Returns an integer between 0 and self.out_size """
        return torch.argmax(self(x, task), dim=1)

    def reset_for_new_task(self, head_idx):
        for layer in self.shared_layers:
            if isinstance(layer, VariationalLayer):
                layer.reset_for_next_task()

        if isinstance(self.heads[head_idx], VariationalLayer):
            self.heads[head_idx].reset_for_next_task()

    def get_statistics(self) -> (list, dict):
        layer_statistics = []
        model_statistics = {
            'average_w_mean': 0,
            'average_b_mean': 0,
            'average_w_var': 0,
            'average_b_var': 0
        }

        n_layers = 0
        for layer in self.shared_layers:
            n_layers += 1
            layer_statistics.append(layer.get_statistics())
            model_statistics['average_w_mean'] += layer_statistics[-1]['average_w_mean']
            model_statistics['average_b_mean'] += layer_statistics[-1]['average_b_mean']
            model_statistics['average_w_var'] += layer_statistics[-1]['average_w_var']
            model_statistics['average_b_var'] += layer_statistics[-1]['average_b_var']

        for head in self.heads:
            n_layers += 1
            layer_statistics.append(head.get_statistics())
            model_statistics['average_w_mean'] += layer_statistics[-1]['average_w_mean']
            model_statistics['average_b_mean'] += layer_statistics[-1]['average_b_mean']
            model_statistics['average_w_var'] += layer_statistics[-1]['average_w_var']
            model_statistics['average_b_var'] += layer_statistics[-1]['average_b_var']

        # todo averaging averages like this is actually incorrect (assumes equal num of params in each layer)
        model_statistics['average_w_mean'] /= n_layers
        model_statistics['average_b_mean'] /= n_layers
        model_statistics['average_w_var'] /= n_layers
        model_statistics['average_b_var'] /= n_layers

        return layer_statistics, model_statistics

    def _kl_divergence(self, head_idx):
        kl_divergence = torch.zeros(1, requires_grad=False).to(self.device)

        # since we are assuming that all parameters are normally distributed independently
        # of each other, the KL divergence formula breaks down into summation of the KL
        # divergences for each parameter
        for layer in self.shared_layers:
            kl_divergence = torch.add(kl_divergence, layer.kl_divergence())

        kl_divergence = torch.add(kl_divergence, self.heads[head_idx].kl_divergence())

        return kl_divergence


class GenerativeVCL(VCL):
    """
    A Bayesian neural network which is updated using variational inference
    methods, and which is multi-headed at the input end. Suitable for
    continual learning of generative tasks.
    """

    def __init__(self, z_dim, h_dim, x_dim, n_heads=0, shared_h_dims=(500,), head_h_dims=(500,),
                 initial_posterior_variance=1e-6):
        super().__init__()
        # dimensions
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.x_dim = x_dim
        self.n_heads = n_heads
        self.ipv = initial_posterior_variance

        shared_dims = [x_dim] + list(shared_h_dims) + [h_dim]
        head_dims = [h_dim] + list(head_h_dims) + [z_dim]

        # layers in task-specific input heads
        if n_heads > 1:
            self.heads = nn.ModuleList([
                [MeanFieldGaussianLinear(head_dims[i], head_dims[i + 1], self.ipv) for i in range(len(head_dims) - 1)]
                for _ in
                range(n_heads)
            ])
        # list of layers in shared network
        self.shared_layers = nn.ModuleList([
            MeanFieldGaussianLinear(shared_dims[i], shared_dims[i + 1], self.ipv) for i in range(len(shared_dims) - 1)
        ])

    def forward(self, x, head_idx, sample_parameters=True):
        if self.n_heads > 0:
            head = self.heads[head_idx]

            for layer in head:
                x = F.relu(layer(x, sample_parameters=sample_parameters))

        for layer in self.shared_layers:
            x = F.relu(layer(x, sample_parameters=sample_parameters))

        return x

    def vcl_loss(self, x, y, head_idx, task_size):
        pass

    def point_estimate_loss(self, x, y, head_idx):
        pass

    def reset_for_new_task(self, head_idx):
        pass

    def get_statistics(self) -> dict:
        pass

    def _kl_divergence(self, head_idx) -> float:
        pass

    def _log_likelihood(self, x, y, head_idx) -> float:
        pass
