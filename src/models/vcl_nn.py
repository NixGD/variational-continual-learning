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
        self.init_variables()
        self.task_id = 0

    def forward(self, x):
        (w_means, w_vars), (b_means, b_vars) = self.posterior
        sampled_weights, sampled_bias = [], []
        for i in range(self.n_hidden_layers + 1):
            w_epsilons = torch.randn_like(w_means[i])
            b_epsilons = torch.randn_like(b_means[i])

            sampled_weights.append(w_means[i] + w_epsilons * torch.sqrt(w_vars[i]))
            sampled_bias.append(b_means[i] + b_epsilons * torch.sqrt(b_vars[i]))

        # Apply each layer
        for weight, bias in zip(sampled_weights, sampled_bias):
            x = F.relu(x @ weight + bias)

        # Apply final softmax
        sm = torch.nn.Softmax(dim=1)
        x = sm(x)

        return x

    def prediction(self):
        pass

    def calculate_KL_term(self):
        """
        Calculates and returns the KL divergence of the new posterior and the previous
        iteration's posterior. See equation L3, slide 14.
        """
        # Concatenate w and b statistics into one tensor for ease of calculation
        ((prior_w_means, prior_w_vars), (prior_b_means, prior_b_vars)) = self.prior
        prior_means = torch.cat([torch.reshape(prior_w_means[i], (-1,)) for i in range(self.n_hidden_layers + 1)] +
                                prior_b_means)
        prior_vars  = torch.cat([torch.reshape(prior_w_vars[i], (-1,)) for i in range(self.n_hidden_layers + 1)] +
                                prior_b_vars)

        ((post_w_means, post_w_vars), (post_b_means, post_b_vars)) = self.posterior
        post_means = torch.cat([torch.reshape(post_w_means[i], (-1,)) for i in range(self.n_hidden_layers + 1)] +
                                post_b_means)
        post_vars  = torch.cat([torch.reshape(post_w_vars[i], (-1,)) for i in range(self.n_hidden_layers + 1)] +
                                post_b_vars)

        # Calculate KL for individual normal distributions over parameters
        KL_elementwise = \
            post_vars / prior_vars + \
            torch.pow(prior_means - post_means, 2) / prior_vars \
            - 1 + torch.log(prior_vars / post_vars)

        # Sum KL over all parameters
        return 0.5 * KL_elementwise.sum()

    def logprob(self, x, y):
        preds = self.forward(x)

        # Make mask to select probabilities associated with actual y values
        mask = torch.zeros(preds.size(), dtype=torch.uint8)
        for i in range(preds.size()[0]):
            mask[i][y[i].item()] = 1
        
        # Select probabilities, log and sum them
        y_preds = torch.masked_select(preds, mask)
        return torch.sum(torch.log(y_preds))

    def loss(self, x, y):
        return self.calculate_KL_term() - self.logprob(x, y)

    def reset_for_new_task(self):
        """
        Called after completion of a task, to reset state for the next task
        """
        self.task_id += 1

        # Set the value of the prior to be the current value of the posterior
        (prior_w_means, prior_w_vars), (prior_b_means, prior_b_vars) = self.prior
        (posterior_w_means, posterior_w_vars), (posterior_b_means, posterior_b_vars) = self.prior
        prior_w_means.data.copy_(posterior_w_means.data)
        prior_w_vars.data.copy_(posterior_w_vars.data)
        prior_b_means.data.copy_(posterior_b_means.data)
        prior_b_vars.data.copy_(posterior_b_vars.data)

    def init_variables(self):
        """
        Called once, on model creation, to set up the prior and posterior
        tensors and set them to their initial values
        """
        # The first prior is initialised to be zero mean, unit variance
        prior_w_means = [torch.zeros(self.input_size, self.layer_width)] + \
                        [torch.zeros(self.layer_width, self.layer_width) for i in range(self.n_hidden_layers - 1)] + \
                        [torch.zeros(self.layer_width, self.out_size)]
        prior_w_vars  = [torch.ones(self.input_size, self.layer_width)] + \
                        [torch.ones(self.layer_width, self.layer_width) for i in range(self.n_hidden_layers - 1)] + \
                        [torch.ones(self.layer_width, self.out_size)]
        prior_b_means = [torch.zeros(self.layer_width) for i in range(self.n_hidden_layers)] + \
                        [torch.zeros(self.out_size)]
        prior_b_vars  = [torch.ones(self.layer_width) for i in range(self.n_hidden_layers)] + \
                        [torch.ones(self.out_size)]

        self.prior = ((prior_w_means, prior_w_vars), (prior_b_means, prior_b_vars))

        # Prior tensors are registered as buffers to indicate to PyTorch that
        # they are persistent state but shouldn't be updated by the optimiser
        for i in range(self.n_hidden_layers + 1):
            self.register_buffer("prior_w_means_" + str(i), prior_w_means[i])
            self.register_buffer("prior_w_vars_" + str(i), prior_w_vars[i])
            self.register_buffer("prior_b_means_" + str(i), prior_b_means[i])
            self.register_buffer("prior_b_vars_" + str(i), prior_b_vars[i])

        # The first posterior is initialised to be the same as the first prior
        posterior_w_means, posterior_w_vars, posterior_b_means, posterior_b_vars = [], [], [], []
        for i in range(self.n_hidden_layers + 1):
            posterior_w_means.append(nn.Parameter(prior_w_means[i].clone().detach().requires_grad_(True)))
            posterior_w_vars.append(nn.Parameter(prior_w_vars[i].clone().detach().requires_grad_(True)))
            posterior_b_means.append(nn.Parameter(prior_b_means[i].clone().detach().requires_grad_(True)))
            posterior_b_vars.append(nn.Parameter(prior_b_vars[i].clone().detach().requires_grad_(True)))

        self.posterior = ((posterior_w_means, posterior_w_vars), (posterior_b_means, posterior_b_vars))

        # Posterior tensors are registered as parameters to indicate to PyTorch
        # that they are persistent state that should be updated by the optimiser
        for i in range(self.n_hidden_layers + 1):
            self.register_parameter("posterior_w_means_" + str(i), posterior_w_means[i])
            self.register_parameter("posterior_w_vars_" + str(i), posterior_w_vars[i])
            self.register_parameter("posterior_b_means_" + str(i), posterior_b_means[i])
            self.register_parameter("posterior_b_vars_" + str(i), posterior_b_vars[i])
