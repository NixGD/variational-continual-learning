import torch
import torch.nn as nn
import torch.nn.functional as F
from util.operations import concatenate_flattened
from layers.distributional import DistributionalLinear
import math


EPSILON = 1e-8  # Small value to avoid divide-by-zero and log(zero) problems


class DiscriminativeVCL(nn.Module):
    """
    A Bayesian multi-head neural network which updates its parameters using
    variational inference.
    """

    def __init__(self, in_size: int, out_size: int, layer_width: int,
                 n_hidden_layers: int, n_tasks: int, initial_posterior_var: int):
        super().__init__()
        self.input_size = in_size
        self.out_size = out_size
        self.n_hidden_layers = n_hidden_layers
        self.layer_width = layer_width
        self.n_tasks = n_tasks

        self.prior, self.posterior = None, None
        self.head_prior, self.head_posterior = None, None

        self._init_variables(initial_posterior_var)

    def to(self, *args, **kwargs):
        """
        Our prior tensors are registered as buffers but the way we access them
        indirectly (through tuple attributes on the model) is causing problems
        because when we use `.to()` to move the model to a new device, the prior
        tensors get moved (because they're registered as buffers) but the
        references in the tuples don't get updated to point to the new moved
        tensors. This has no effect when running just on a cpu but breaks the
        model when trying to run on a gpu. There are a million nicer ways of
        working around this problem, but for now the easiest thing is to do
        this: override the `.to()` method and manually update our references to
        prior tensors.
        """
        self = super().to(*args, **kwargs)
        (prior_w_means, prior_w_log_vars), (prior_b_means, prior_b_log_vars) = self.prior
        prior_w_means = [t.to(*args, **kwargs) for t in prior_w_means]
        prior_w_log_vars = [t.to(*args, **kwargs) for t in prior_w_log_vars]
        prior_b_means = [t.to(*args, **kwargs) for t in prior_b_means]
        prior_b_log_vars = [t.to(*args, **kwargs) for t in prior_b_log_vars]
        self.prior = (prior_w_means, prior_w_log_vars), (prior_b_means, prior_b_log_vars)
        (head_prior_w_means, head_prior_w_log_vars), (head_prior_b_means, head_prior_b_log_vars) = self.head_prior
        head_prior_w_means = [t.to(*args, **kwargs) for t in head_prior_w_means]
        head_prior_w_log_vars = [t.to(*args, **kwargs) for t in head_prior_w_log_vars]
        head_prior_b_means = [t.to(*args, **kwargs) for t in head_prior_b_means]
        head_prior_b_log_vars = [t.to(*args, **kwargs) for t in head_prior_b_log_vars]
        self.head_prior = (head_prior_w_means, head_prior_w_log_vars), (head_prior_b_means, head_prior_b_log_vars)
        return self

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
        x = torch.nn.Softmax(dim=1)(x)

        return x

    def vcl_loss(self, x, y, head, task_size) -> torch.Tensor:
        """
        Returns the loss of the model, as described in equation 4 of the Variational
        Continual Learning paper (https://arxiv.org/abs/1710.10628).
        """
        return self._calculate_kl_term().cpu() / task_size - self._log_prob(x, y, head)

    def point_estimate_loss(self, x, y, task=0):
        """
        Returns a loss defined in terms of a simplified forward pass that
        doesn't use sampling, and so uses the posterior means but not the
        variances. Used as part of model initialisation to optimise the
        posterior means to point-estimates for the first task.
        """
        (w_means, _), (b_means, _) = self.posterior
        (head_w_means, _), (head_b_means, _) = self.head_posterior

        for weight, bias in zip(w_means, b_means):
            x = F.relu(x @ weight + bias)

        x = x @ head_w_means[task] + head_b_means[task]

        return nn.CrossEntropyLoss()(x, y)

    def prediction(self, x, task):
        """ Returns an integer between 0 and self.out_size """
        return torch.argmax(self.forward(x, task), dim=1)

    def reset_for_new_task(self, head):
        """
        Called after completion of a task, to reset state for the next task
        """
        # Set the value of the prior to be the current value of the posterior
        (prior_w_means, prior_w_log_vars), (prior_b_means, prior_b_log_vars) = self.prior
        (post_w_means, post_w_log_vars), (post_b_means, post_b_log_vars) = self.posterior
        for i in range(self.n_hidden_layers):
            prior_w_means[i].data.copy_(post_w_means[i].data)
            prior_w_log_vars[i].data.copy_(post_w_log_vars[i].data)
            prior_b_means[i].data.copy_(post_b_means[i].data)
            prior_b_log_vars[i].data.copy_(post_b_log_vars[i].data)

        # set the value of the head prior to be the current value of the posterior
        (head_prior_w_means, head_prior_w_log_vars), (head_prior_b_means, head_prior_b_log_vars) = self.head_prior
        (head_posterior_w_means, head_posterior_w_log_vars), (head_posterior_b_means, head_posterior_b_log_vars) = self.head_posterior
        head_prior_w_means[head].data.copy_(head_posterior_w_means[head].data)
        head_prior_w_log_vars[head].data.copy_(head_posterior_w_log_vars[head].data)
        head_prior_b_means[head].data.copy_(head_posterior_b_means[head].data)
        head_prior_b_log_vars[head].data.copy_(head_posterior_b_log_vars[head].data)

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

    def _log_prob(self, x, y, head):
        predictions = self.forward(x, head)

        # Make mask to select probabilities associated with actual y values
        mask = torch.zeros(predictions.size(), dtype=torch.uint8)
        for i in range(predictions.size()[0]):
            mask[i][int(y[i].item())] = 1

        # Select probabilities, log and sum them
        y_preds = torch.masked_select(predictions.cpu(), mask)
        return torch.mean(torch.log(y_preds + EPSILON))

    def _sample_parameters(self, w_means, b_means, w_log_vars, b_log_vars):
        # sample weights and biases from normal distributions
        sampled_weights, sampled_bias = [], []
        for layer_n in range(len(w_means)):
            w_epsilons = torch.randn_like(w_means[layer_n])
            b_epsilons = torch.randn_like(b_means[layer_n])
            sampled_weights.append(w_means[layer_n] + w_epsilons * torch.exp(0.5 * w_log_vars[layer_n]))
            sampled_bias.append(b_means[layer_n] + b_epsilons * torch.exp(0.5 * b_log_vars[layer_n]))
        return zip(sampled_weights, sampled_bias)

    def _init_variables(self, initial_posterior_var):
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

        empty_parameter_like = lambda t: nn.Parameter(torch.empty_like(t, requires_grad=True))

        posterior_w_means = [empty_parameter_like(t) for t in prior_w_means]
        posterior_w_log_vars = [empty_parameter_like(t) for t in prior_w_log_vars]
        posterior_b_means = [empty_parameter_like(t) for t in prior_b_means]
        posterior_b_log_vars = [empty_parameter_like(t) for t in prior_b_log_vars]

        self.posterior = ((posterior_w_means, posterior_w_log_vars), (posterior_b_means, posterior_b_log_vars))

        head_posterior_w_means = [empty_parameter_like(t) for t in head_prior_w_means]
        head_posterior_w_log_vars = [empty_parameter_like(t) for t in head_prior_w_log_vars]
        head_posterior_b_means = [empty_parameter_like(t) for t in head_prior_b_means]
        head_posterior_b_log_vars = [empty_parameter_like(t) for t in head_prior_b_log_vars]

        self.head_posterior = \
            ((head_posterior_w_means, head_posterior_w_log_vars),
             (head_posterior_b_means, head_posterior_b_log_vars))

        # Initialise the posterior means with a normal distribution. Note that
        # prior to training we will run a procedure to optimise these values to
        # point-estimates of the parameters for the first task.
        for t in posterior_w_means + posterior_b_means + head_posterior_w_means + head_posterior_b_means:
            torch.nn.init.normal_(t, mean=0, std=1)

        # Initialise the posterior variances with the given constant value.
        for t in posterior_w_log_vars + posterior_b_log_vars + head_posterior_w_log_vars + head_posterior_b_log_vars:
            torch.nn.init.constant_(t, math.log(initial_posterior_var))

        # Finally, we register the prior and the posterior with the nn.Module.
        # The prior values are registered as buffers, which indicates to PyTorch
        # that they represent persistent state which should not be updated by
        # the optimizer. The posteriors are registered as parameters, which on
        # the other hand are to be modified by the optimizer.
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

    def _mean_posterior_variance(self):
        """
        Return the mean posterior variance for logging purposes.
        Excludes the head layer.
        """
        ((_, posterior_w_log_vars), (_, posterior_b_log_vars)) = self.posterior
        posterior_log_vars = torch.cat([torch.reshape(t, (-1,)) for t in posterior_w_log_vars] + posterior_b_log_vars)
        posterior_vars     = torch.exp(posterior_log_vars)
        return torch.mean(posterior_vars).item()

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
