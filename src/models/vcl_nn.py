import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class VCL_NN(nn.Module):
    """A Bayesian multi-head neural network which updates its parameters using
       variational inference."""

    def __init__(self, input_size: int, out_size: int, layer_width: int, n_hidden_layers: int):
        super(VCL_NN, self).__init__()
        self.input_size = input_size
        self.out_size = out_size
        self.n_hidden_layers = n_hidden_layers
        self.layer_width = layer_width
        self.prior, self.posterior = None, None
        self.init_prior()

    def forward(self, x):
        (w_means, w_vars), (b_means, b_vars) = self.posterior
        w_epsilons = torch.randn_like(w_means)
        b_epsilons = torch.randn_like(b_means)

        sampled_weights = w_means + w_epsilons * torch.sqrt(w_vars)
        sampled_bias = b_means + b_epsilons * torch.sqrt(b_vars)

        for i, layer in enumerate(sampled_weights):
            x = F.relu(layer @ x + sampled_bias[i])

        return x

    def prediction(self):
        pass

    def calculate_KL_term(self):
        '''Calculates and returns KL(posterior, prior). Formula from L3 slide 14.'''
        # Concatenate w and b statistics into one tensor for ease of calculation
        ((prior_w_means, prior_w_vars), (prior_b_means, prior_b_vars)) = self.prior
        prior_means = torch.cat((prior_w_means, prior_b_means), axis=0)
        prior_vars = torch.cat((prior_w_vars, prior_b_vars), axis=0)

        ((post_w_means, post_w_vars), (post_b_means, post_b_vars)) = self.posterior
        post_means = torch.cat((post_w_means, post_b_means), axis=0)
        post_vars = torch.cat((post_w_vars, post_b_vars), axis=0)

        # Calculate KL for individual normal distributions over parameters
        KL_elementwise = \
            post_vars / prior_vars + \
            torch.pow(prior_means - post_means, 2) / prior_vars \
            - 1 + torch.log(prior_vars / post_vars)

        # Sum KL over all parameters
        return 0.5 * KL_elementwise.sum()

    def loss(self):
        pass

    def init_prior(self):
        if self.prior == None:
            w_means = torch.zeros(self.n_hidden_layers + 1,
                                  self.layer_width, self.layer_width)
            w_vars = torch.ones(self.n_hidden_layers + 1,
                                self.layer_width, self.layer_width)
            b_means = torch.zeros(self.n_hidden_layers + 1, self.layer_width)
            b_vars = torch.ones(self.n_hidden_layers + 1, self.layer_width)
            self.prior = ((w_means, w_vars), (b_means, b_vars))
        else:
            self.prior = self.posterior
