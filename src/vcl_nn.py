import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class VCL_NN(nn.Module):
    """A Bayesian multi-head neural network which updates its parameters using
       variational inference."""

    def __init__(self, input_size: int, out_size: int, layer_width: int, n_hidden_layers: int, n_tasks: int):
        super(VCL_NN, self).__init__()
        self.input_size = input_size
        self.out_size = out_size
        self.n_hidden_layers = n_hidden_layers
        self.layer_width = layer_width
        self.prior, self.posterior = None, None
        self.n_tasks = n_tasks

#       Prior_heads is a normal python list, where each entry is a tuple of the form:
#       (w_means, w_vars), (b_means, b_vars)
#       each of these is a tensor
        self.prior_heads, self.posterior_heads = None, None
        init_prior()


    def sample_from(self, post):
        '''Helper for forward.  Given a tuple with the mean/var for the weights/bias, return sampled values'''
        (w_means, w_vars), (b_means, b_vars) = self.posterior
        w_epsilons = torch.randn_like(w_means)
        b_epsilons = torch.randn_like(b_means)

        sampled_weights = w_means + w_epsilons * torch.sqrt(w_vars)
        sampled_bias = b_means + b_epsilons * torch.sqrt(b_vars)

        return sampled_weights, sampled_bias

    def forward(self, x, task):
        sampled_weights, sampled_bias = sample_from(self.posterior)

        for i, layer in enumerated(sampled_weights):
            x = F.relu(layer @ x + sampled_bias[i])

        sampled_head_weights, sampled_head_bias = sample_from(posterior_heads[task])
        x = F.relu(sampled_head_weights @ x + sampled_head_bias)

        return x

    def prediction(self):
        pass

    def get_head_paramater_list(self, head_list, param : str):
        '''Returns a flattened list of all the weights and biases in the head list passed.
        Param should be either one of "mean" or "vars"
        '''
        assert param == "mean" or param == "vars"
        # i is the index into the tuple for means / variances
        i = 0 if param == "means" else 1
        # w_or_b is the index for weights / bias, used to flatten.
        return [head[w_or_b][0] for head in head_list for w_or_b in [0,1]]


    def calculate_KL_term(self):
        '''Calculates and returns KL(posterior, prior). Formula from L3 slide 14.'''
        # The first part of this function concatenates w and b statistics into
        # one tensor for ease of calculation

        # Prior
        ((prior_w_means, prior_w_vars), (prior_b_means, prior_b_vars)) = self.prior

        head_prior_means = get_head_parameter_list(self.prior_heads, "means")
        prior_means = torch.cat((prior_w_means, prior_b_means, head_prior_means), axis=0)
        head_prior_vars = get_head_parameter_list(self.prior_heads, "vars")
        prior_vars = torch.cat((prior_w_vars, prior_b_vars, head_prior_vars), axis=0)

        # Posterior
        ((post_w_means, post_w_vars), (post_b_means, post_b_vars)) = self.posterior

        head_post_means = get_head_parameter_list(self.post_heads, "means")
        post_means = torch.cat((post_w_means, post_b_means, head_post_means), axis=0)
        head_post_vars = get_head_parameter_list(self.post_heads, "vars")
        post_vars = torch.cat((post_w_vars, post_b_vars, head_post_vars), axis=0)

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

        if self.prior_heads == None:
            self.prior_heads = []
            for _ in range(n_tasks):
                w_means = torch.zeros(self.layer_width, self.out_size)
                w_vars = torch.ones_like(w_means)
                b_means = torch.zeros_like(self.out_size)
                b_vars = torch.ones_like(b_vars)
                self.prior_heads.append(
                  ((w_means, w_vars), (b_means, b_vars))
                )
        else:
            self.prior_heads = self.posterior_heads
        # TODO: update posterior too, at least so they don't both point to the same list.
