import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.init_variables()
        self.task_id = 0

    def sample_from(self, means, vars):
        '''Helper for forward.  Given a tuple with the mean/var for the weights/bias, return sampled values'''
        epsilons = torch.randn_like(vars)
        return means + epsilons * torch.sqrt(vars)

    def forward(self, x):
        (w_means, w_vars), (b_means, b_vars) = self.posterior
        sampled_weights = [self.sample_from(w_means[i], w_vars[i]) for i in range(self.n_hidden_layers + 1)]
        sampled_bias = [self.sample_from(b_means[i], b_vars[i]) for i in range(self.n_hidden_layers + 1)]

        # Apply each layer
        for weight, bias in zip(sampled_weights, sampled_bias):
            x = F.relu(x @ weight + bias)

        (head_w_mean, head_w_vars), (head_b_means, head_b_vars) = self.posterior_heads[task]
        sampled_head_weights = self.sample_from(head_w_mean, head_w_vars)
        sampled_head_bias = self.sample_from(head_b_means, head_b_vars)

        x = F.relu(x @ sampled_head_weights + sampled_head_bias)

        sm = torch.nn.Softmax(dim=1)
        x = sm(x)

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
        prior_means = torch.cat([torch.reshape(prior_w_means[i], (-1,)) for i in range(self.n_hidden_layers + 1)] +
                                prior_b_means +
                                [self.get_head_parameter_list(self.prior_heads, "means")])
        prior_vars  = torch.cat([torch.reshape(prior_w_vars[i], (-1,)) for i in range(self.n_hidden_layers + 1)] +
                                prior_b_vars +
                                [self.get_head_parameter_list(self.prior_heads, "vars")])

        # Posterior
        ((post_w_means, post_w_vars), (post_b_means, post_b_vars)) = self.posterior
        post_means = torch.cat([torch.reshape(post_w_means[i], (-1,)) for i in range(self.n_hidden_layers + 1)] +
                                post_b_means +
                                [self.get_head_parameter_list(self.post_heads, "means")])
        post_vars  = torch.cat([torch.reshape(post_w_vars[i], (-1,)) for i in range(self.n_hidden_layers + 1)] +
                                post_b_vars +
                                [self.get_head_parameter_list(self.post_heads, "vars")])

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

        # Update heads too.

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


        # TODO: I'm confused about what's happening above, so this is almost certainly going to need fixing.
        # I just want to get the merge through first though.

        if self.prior_heads == None:
            self.prior_heads = []
            for _ in range(self.n_tasks):
                w_means = torch.zeros(self.layer_width, self.out_size)
                w_vars = torch.ones_like(w_means)
                b_means = torch.zeros_like(self.out_size)
                b_vars = torch.ones_like(b_vars)
                self.prior_heads.append(
                  ((w_means, w_vars), (b_means, b_vars))
                )
        else:
            self.prior_heads = self.posterior_heads
