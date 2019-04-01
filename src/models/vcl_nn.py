import torch
import torch.nn as nn
import torch.nn.functional as F
from util.operations import concatenate_flattened
from layers.distributional import DistributionalLinear


EPSILON = 1e-8  # Small value to avoid divide-by-zero and log(zero) problems


class DiscriminativeVCL(nn.Module):
    """
    A Bayesian multi-head neural network which updates its parameters using
    variational inference.
    """

    def __init__(self, in_size: int, out_size: int, layer_width: int, n_hidden_layers: int, n_tasks: int):
        super().__init__()
        self.input_size = in_size
        self.out_size = out_size
        self.n_hidden_layers = n_hidden_layers
        self.layer_width = layer_width
        self.n_tasks = n_tasks

        self.prior, self.posterior = None, None
        self.head_prior, self.head_posterior = None, None

        self._init_variables()
        self.task_id = 0

    def forward(self, x, task):
        """ Forward pass of the model on an input. """
        # sample layer parameters from posterior distribution
        (w_means, w_log_vars), (b_means, b_log_vars) = self.posterior
        (head_w_means, head_w_log_vars), (head_b_means, head_b_log_vars) = self.head_posterior
        sampled_layers = self._sample_parameters(w_means, b_means, w_log_vars, b_log_vars)
        sampled_head_layers = self._sample_parameters(head_w_means, head_b_means, head_w_log_vars, head_b_log_vars)

        # Apply each layer with its sampled weights and biases
        for weight, bias in sampled_layers:
            x = F.relu(x @ weight + bias)

        head_weight, head_bias = list(sampled_head_layers)[task]
        x = x @ head_weight + head_bias

        # Apply final softmax
        sm = torch.nn.Softmax(dim=1)
        x = sm(x)

        return x

    def loss(self, x, y, task) -> torch.Tensor:
        """
        Returns the loss of the model, as described in equation 4 of the Variational
        Continual Learning paper (https://arxiv.org/abs/1710.10628).
        """
        return self._calculate_kl_term() - self._log_prob(x, y, task)

    def prediction(self, x, task):
        """ Returns an integer between 0 and self.out_size """
        return torch.argmax(self.forward(x, task), dim=1)

    def reset_for_new_task(self):
        """
        Called after completion of a task, to reset state for the next task
        """
        self.task_id += 1

        # Set the value of the prior to be the current value of the posterior
        (prior_w_means, prior_w_log_vars), (prior_b_means, prior_b_log_vars) = self.prior
        (post_w_means, post_w_log_vars), (post_b_means, post_b_log_vars) = self.posterior
        prior_w_means.data.copy_(post_w_means.data)
        prior_w_log_vars.data.copy_(post_w_log_vars.data)
        prior_b_means.data.copy_(post_b_means.data)
        prior_b_log_vars.data.copy_(post_b_log_vars.data)

        # set the value of the head prior to be the current value of the posterior
        (head_prior_w_means, head_prior_w_log_vars), (head_prior_b_means, head_prior_b_log_vars) = self.head_prior
        (head_posterior_w_means, head_posterior_w_log_vars), (head_posterior_b_means, head_posterior_b_log_vars) = self.head_posterior
        head_prior_w_means.data.copy_(head_posterior_w_means.data)
        head_prior_w_log_vars.data.copy_(head_posterior_w_log_vars.data)
        head_prior_b_means.data.copy_(head_posterior_b_means.data)
        head_prior_b_log_vars.data.copy_(head_posterior_b_log_vars.data)

    def _calculate_kl_term(self):
        """
        Calculates and returns the KL divergence of the new posterior and the previous
        iteration's posterior. See equation L3, slide 14.
        """
        # Prior
        ((prior_w_means, prior_w_log_vars), (prior_b_means, prior_b_log_vars)) = self.prior
        ((head_prior_w_means, head_prior_w_log_vars),
         (head_prior_b_means, head_prior_b_log_vars)) = self.head_prior

        prior_means = concatenate_flattened(
            prior_w_means + head_prior_w_means +
            prior_b_means + head_prior_b_means)
        prior_log_vars = concatenate_flattened(
            prior_w_log_vars + head_prior_w_log_vars +
            prior_b_log_vars + head_prior_b_log_vars)
        prior_vars = torch.exp(prior_log_vars)

        # Posterior
        ((post_w_means, post_w_log_vars), (post_b_means, post_b_log_vars)) = self.posterior
        ((head_post_w_means, head_post_w_log_vars),
         (head_post_b_means, head_post_b_log_vars)) = self.head_posterior

        post_means = concatenate_flattened(
            post_w_means + head_post_w_means +
            post_b_means + head_post_b_means)
        post_log_vars = concatenate_flattened(
            post_w_log_vars + head_post_w_log_vars +
            post_b_log_vars + head_post_b_log_vars)
        post_vars = torch.exp(post_log_vars)

        # Calculate KL for individual normal distributions over parameters
        kl_elementwise = \
            post_vars / (prior_vars + EPSILON) + \
            torch.pow(prior_means - post_means, 2) / (prior_vars + EPSILON) \
            - 1 + prior_log_vars - post_log_vars

        # Sum KL over all parameters
        return 0.5 * kl_elementwise.sum()

    def _log_prob(self, x, y, task):
        predictions = self.forward(x, task)

        # Make mask to select probabilities associated with actual y values
        mask = torch.zeros(predictions.size(), dtype=torch.uint8)
        for i in range(predictions.size()[0]):
            mask[i][int(y[i].item())] = 1

        # Select probabilities, log and sum them
        y_preds = torch.masked_select(predictions, mask)
        return torch.sum(torch.log(y_preds + EPSILON))

    def _sample_parameters(self, w_means, b_means, w_log_vars, b_log_vars):
        # sample weights and biases from normal distributions
        sampled_weights, sampled_bias = [], []
        for layer_n in range(len(w_means)):
            w_epsilons = torch.randn_like(w_means[layer_n])
            b_epsilons = torch.randn_like(b_means[layer_n])
            sampled_weights.append(w_means[layer_n] + w_epsilons * torch.exp(0.5 * w_log_vars[layer_n]))
            sampled_bias.append(b_means[layer_n] + b_epsilons * torch.exp(0.5 * b_log_vars[layer_n]))
        return zip(sampled_weights, sampled_bias)

    def _init_variables(self):
        """
        Initializes the model's prior and posterior weights / biases to their initial
        values. This method is called once on model creation. The model prior is registered
        as a persistent part of the model state which should not be modified, while the
        initial posterior is registered as a model parameter to be optimized.

        To avoid negative variances, we do not store parameter variances directly; instead
        we store the logarithm of each variance, and apply the exponential as needed in the
        forward pass.
        """
        # The initial prior over the parameters has zero mean, unit variance (i.e. log variance 0)
        prior_w_means = [torch.zeros(self.input_size, self.layer_width)] + \
                        [torch.zeros(self.layer_width, self.layer_width) for _ in range(self.n_hidden_layers - 1)]
        prior_w_log_vars = [torch.zeros_like(t) for t in prior_w_means]
        prior_b_means = [torch.zeros(self.layer_width) for _ in range(self.n_hidden_layers)]
        prior_b_log_vars = [torch.zeros_like(t) for t in prior_b_means]

        self.prior = ((prior_w_means, prior_w_log_vars), (prior_b_means, prior_b_log_vars))

        head_prior_w_means = [torch.zeros(self.layer_width, self.out_size) for t in range(self.n_tasks)]
        head_prior_w_log_vars = [torch.zeros_like(t) for t in head_prior_w_means]
        head_prior_b_means = [torch.zeros(self.out_size) for t in range(self.n_tasks)]
        head_prior_b_log_vars = [torch.zeros_like(t) for t in head_prior_b_means]

        self.head_prior = ((head_prior_w_means, head_prior_w_log_vars), (head_prior_b_means, head_prior_b_log_vars))

        # The initial posterior is initialised to be the same as the first prior
        grad_copy = lambda t: nn.Parameter(t.clone().detach().requires_grad_(True))

        posterior_w_means = [grad_copy(t) for t in prior_w_means]
        posterior_w_log_vars = [grad_copy(t) for t in prior_w_log_vars]
        posterior_b_means = [grad_copy(t) for t in prior_b_means]
        posterior_b_log_vars = [grad_copy(t) for t in prior_b_log_vars]

        self.posterior = ((posterior_w_means, posterior_w_log_vars), (posterior_b_means, posterior_b_log_vars))

        head_posterior_w_means = [grad_copy(t) for t in head_prior_w_means]
        head_posterior_w_log_vars = [grad_copy(t) for t in head_prior_w_log_vars]
        head_posterior_b_means = [grad_copy(t) for t in head_prior_b_means]
        head_posterior_b_log_vars = [grad_copy(t) for t in head_prior_b_log_vars]

        self.head_posterior = \
            ((head_posterior_w_means, head_posterior_w_log_vars),
             (head_posterior_b_means, head_posterior_b_log_vars))

        # finally, we register the prior and the posterior with the nn.Module. The
        # prior values are registered as buffers, which indicates to PyTorch that they
        # represent persistent state which should not be updated by the optimizer. The
        # posteriors are registered as parameters, which on the other hand are to
        # be modified by the optimizer.
        for i in range(self.n_hidden_layers):
            self.register_buffer("prior_w_means_" + str(i), prior_w_means[i])
            self.register_buffer("prior_w_log_vars_" + str(i), prior_w_log_vars[i])
            self.register_buffer("prior_b_means_" + str(i), prior_b_means[i])
            self.register_buffer("prior_b_log_vars_" + str(i), prior_b_log_vars[i])

        for i in range(self.n_tasks):
            self.register_buffer("head_prior_w_means_" + str(i), head_prior_w_means[i])
            self.register_buffer("head_prior_w_log_vars_" + str(i), head_prior_w_log_vars[i])
            self.register_buffer("head_prior_b_means_" + str(i), head_prior_b_means[i])
            self.register_buffer("head_prior_b_log_vars_" + str(i), head_prior_b_log_vars[i])

        for i in range(self.n_hidden_layers):
            self.register_parameter("posterior_w_means_" + str(i), posterior_w_means[i])
            self.register_parameter("posterior_w_log_vars_" + str(i), posterior_w_log_vars[i])
            self.register_parameter("posterior_b_means_" + str(i), posterior_b_means[i])
            self.register_parameter("posterior_b_log_vars_" + str(i), posterior_b_log_vars[i])

        for i in range(self.n_tasks):
            self.register_parameter("head_posterior_w_means_" + str(i), head_posterior_w_means[i])
            self.register_parameter("head_posterior_w_log_vars_" + str(i), head_posterior_w_log_vars[i])
            self.register_parameter("head_posterior_b_means_" + str(i), head_posterior_b_means[i])
            self.register_parameter("head_posterior_b_log_vars_" + str(i), head_posterior_b_log_vars[i])


class GenerativeVCL(nn.Module):
    """
    A Bayesian neural network which is updated using variational inference
    methods, and which is multi-headed at the input end. Suitable for
    continual learning of generative tasks.
    """

    def __init__(self, z_dim, h_dim, x_dim, n_heads, n_hidden_layers=(1, 1), hidden_dims=(500, 500)):
        super().__init__()
        # dimensions
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.x_dim = x_dim
        self.n_hidden_layers = n_hidden_layers
        self.hidden_widths = hidden_dims
        # layers in task-specific input heads
        self.heads = [{
            'head_linear_1': DistributionalLinear(z_dim, hidden_dims[0]),
            'head_linear_2': DistributionalLinear(hidden_dims[1], h_dim)
        } for _ in range(n_heads)]
        # layers in shared part of network
        self.shared_linear_1 = DistributionalLinear(h_dim, hidden_dims[1])
        self.shared_linear_2 = DistributionalLinear(hidden_dims[1], x_dim)

    def forward(self, x, head_idx):
        head_linear_1 = self.heads[head_idx]['head_linear_1']
        head_linear_2 = self.heads[head_idx]['head_linear_2']

        x = F.relu(head_linear_1(x))
        x = F.relu(head_linear_2(x))
        x = F.relu(self.shared_linear_1(x))
        x = F.relu(self.shared_linear_2(x))

        return x

    def loss(self):
        # todo
        pass
