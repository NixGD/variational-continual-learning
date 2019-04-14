"""
This module contains the less thoroughly tested implementations of VCL models.

In particular, the models in this module are defined in a different manner to the main
models in the models.vcl_nn module. The models in this module are defined in terms of
bayesian layers from the layers.variational module, which abstract the details of
online variational inference. This approach is in line with the standard style in which
PyTorch models are defined.
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.variational import VariationalLayer, MeanFieldGaussianLinear
from models.deep_models import Encoder
from util.operations import kl_divergence, bernoulli_log_likelihood

EPSILON = 1e-8  # Small value to avoid divide-by-zero and log(zero) problems


class VCL(nn.Module, ABC):
    """ Base class for all VCL models """
    def __init__(self, epsilon=EPSILON):
        super().__init__()
        self.epsilon = epsilon

    @abstractmethod
    def reset_for_new_task(self, head_idx):
        pass

    @abstractmethod
    def get_statistics(self) -> (list, dict):
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
            MeanFieldGaussianLinear(shared_dims[i], shared_dims[i + 1], self.ipv, EPSILON) for i in
            range(len(shared_dims) - 1)
        ])
        # list of heads, each head is a list of layers
        self.heads = nn.ModuleList([
            MeanFieldGaussianLinear(self.h_dim, self.y_dim, self.ipv, EPSILON) for _ in range(n_heads)
        ])

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, head_idx, sample_parameters=True):
        y_out = torch.zeros(size=(x.size()[0], self.y_dim)).to(self.device)

        # repeat forward pass n times to sample layer params multiple times
        for _ in range(self.mc_sampling_n if sample_parameters else 1):
            h = x
            # shared part
            for layer in self.shared_layers:
                h = F.relu(layer(h, sample_parameters=sample_parameters))

            # head
            h = self.heads[head_idx](h, sample_parameters=sample_parameters)
            h = self.softmax(h)

            y_out.add_(h)

        y_out.div_(self.mc_sampling_n)

        return y_out

    def vcl_loss(self, x, y, head_idx, task_size):
        return self._kl_divergence(head_idx) / task_size + torch.nn.NLLLoss()(self(x, head_idx), y)

    def point_estimate_loss(self, x, y, head_idx):
        return torch.nn.NLLLoss()(self(x, head_idx, sample_parameters=False), y)

    def prediction(self, x, task):
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

    def _kl_divergence(self, head_idx) -> torch.Tensor:
        kl_divergence = torch.zeros(1, requires_grad=False).to(self.device)

        # kl divergence is equal to sum of parameter-wise divergences since
        # distribution is diagonal multivariate normal (parameters are independent)
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

    def __init__(self, z_dim, h_dim, x_dim, n_heads=0, encoder_h_dims=(500, 500), decoder_head_h_dims=(500,),
                 decoder_shared_h_dims=(500,), initial_posterior_variance=1e-6, mc_sampling_n=10, device='cpu'):
        super().__init__()
        # handle bad input
        if n_heads < 1:
            raise ValueError('Network requires at least one head.')

        # internal parameters
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.x_dim = x_dim
        self.encoder_h_dims = encoder_h_dims
        self.n_heads = n_heads
        self.ipv = initial_posterior_variance
        self.mc_sampling_n = mc_sampling_n
        self.device = device
        # prior over z
        self.z_prior_mean = 0.0
        self.z_prior_log_variance = 0.0
        # layer dimensions
        head_dims = [z_dim] + list(decoder_head_h_dims) + [h_dim]
        shared_dims = [h_dim] + list(decoder_shared_h_dims) + [x_dim]

        # encoder
        self.encoder = Encoder(x_dim, 2, encoder_h_dims)
        # list of heads, each with a list of layers
        self.decoder_heads = nn.ModuleList([
            nn.ModuleList([MeanFieldGaussianLinear(head_dims[i], head_dims[i + 1], self.ipv) for i in range(len(head_dims) - 1)])
            for _ in
            range(n_heads)
        ])
        # list of layers in shared network
        self.decoder_shared = nn.ModuleList([
            MeanFieldGaussianLinear(shared_dims[i], shared_dims[i + 1], self.ipv, EPSILON) for i in
            range(len(shared_dims) - 1)
        ])

    def forward(self, x, head_idx, sample_parameters=True):
        """ Forward pass for the entire VAE, passing through both the encoder and the decoder. """
        z_params = self.forward_encoder_only(x, head_idx)
        z_means = torch.stack(tuple([z_params[:, 0] for _ in range(self.z_dim)]), dim=1)
        z_variances = torch.stack(tuple([torch.exp(z_params[:, 0]) for _ in range(self.z_dim)]), dim=1)

        z = torch.normal(z_means, z_variances)
        x_out = self.forward_decoder_only(z, head_idx)

        return x_out

    def forward_encoder_only(self, x, head_idx):
        """ Forward pass for the encoder. Takes data as input, and returns the mean and variance
        of the log-normal posterior latent distribution p(z | x), for each data point. This
        distribution is used to sample the actual latent representation z for the data point x. """
        return self.encoder(x, head_idx)

    def forward_decoder_only(self, z, head_idx, sample_parameters=True):
        """ Forward pass for the decoder. Takes a latent representation and produces a reconstruction
        of the original data point that z represents. """
        x_out = torch.zeros(size=(z.size()[0], self.x_dim)).to(self.device)

        # repeat forward pass n times to sample layer params multiple times
        for _ in range(self.mc_sampling_n if sample_parameters else 1):
            h = z
            for layer in self.decoder_heads[head_idx]:
                h = F.relu(layer(h))

            for layer in self.decoder_shared:
                h = F.relu(layer(h))

            x_out.add_(h)
        x_out.div_(self.mc_sampling_n)

        return x_out

    def vae_loss(self, x, task_idx, task_size):
        """ Loss implementing the full variational lower bound from page 5 of the paper """
        elbo = self._elbo(x, task_idx)
        kl = self._kl_divergence(task_idx) / task_size
        return - elbo + kl

    def generate(self, batch_size, task_idx):
        """ Sample new images x from p(x|z)p(z), where z is a gaussian noise distribution. """
        z = torch.randn((batch_size, self.z_dim))

        for layer in self.decoder_heads[task_idx]:
            z = F.relu(layer(z))

        for layer in self.decoder_shared:
            z = F.relu(layer(z))

        return z

    def reset_for_new_task(self, head_idx):
        """ Creates new encoder and resets the decoder (in the VCL sense). """
        self.encoder = Encoder(self.x_dim, self.z_dim, self.encoder_h_dims)
        for layer in self.decoder_shared:
            if isinstance(layer, VariationalLayer):
                layer.reset_for_next_task()

        for layer in self.decoder_heads[head_idx]:
            if isinstance(layer, VariationalLayer):
                layer.reset_for_next_task()

    def get_statistics(self) -> dict:
        pass

    def _elbo(self, x, head_idx, sample_n=1):
        """ Computes the variational lower bound """
        z_params = self.forward_encoder_only(x, head_idx)
        kl = kl_divergence(z_params[:, 0], z_params[:, 1])

        z_means = torch.stack(tuple([z_params[:, 0] for _ in range(self.z_dim)]), dim=1)
        z_variances = torch.stack(tuple([torch.exp(z_params[:, 0]) for _ in range(self.z_dim)]), dim=1)

        log_likelihood = torch.zeros(size=(x.size()[0],)).to(self.device)
        for _ in range(sample_n):
            z = torch.normal(z_means, z_variances)
            x_reconstructed = self.forward_decoder_only(z, head_idx)
            # Bernoulli likelihood of data
            log_likelihood.add_(bernoulli_log_likelihood(x, x_reconstructed, self.epsilon))
        log_likelihood = log_likelihood.div_(sample_n)

        return torch.mean(log_likelihood - kl)

    def _kl_divergence(self, head_idx) -> torch.Tensor:
        """ KL divergence of the VCL decoder's posterior distribution from its previous posterior. """
        kl_divergence = torch.zeros(1, requires_grad=False).to(self.device)

        # kl divergence is equal to sum of parameter-wise divergences since
        # distribution is diagonal multivariate normal (parameters are independent)
        for layer in self.decoder_shared:
            kl_divergence = torch.add(kl_divergence, layer.kl_divergence())

        for layer in self.decoder_heads[head_idx]:
            kl_divergence = torch.add(kl_divergence, layer.kl_divergence())

        return kl_divergence
