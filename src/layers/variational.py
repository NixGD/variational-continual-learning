from abc import ABC, abstractmethod
import math
import torch
import torch.autograd
import torch.nn.init
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch._jit_internal import weak_module, weak_script_method


@weak_module
class VariationalLayer(torch.nn.Module, ABC):
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    @abstractmethod
    def forward(self, x, sample_parameters=True):
        pass

    @abstractmethod
    def reset_for_next_task(self):
        pass

    @abstractmethod
    def kl_divergence(self) -> torch.Tensor:
        pass

    @abstractmethod
    def get_statistics(self) -> dict:
        pass


@weak_module
class MeanFieldGaussianLinear(VariationalLayer):
    """
    A linear transformation on incoming data of the form :math:`y = w^T x + b`,
    where the weights w and the biases b are distributions of parameters
    rather than point estimates. The layer has a prior distribution over its
    parameters, as well as an approximate posterior distribution.

    Args:
        in_features: size of each input sample
        out_features: size of each output sample

    Shape:
        - Input: :math:`(N, *, H_{in})` where :math:`*` means any number of
          additional dimensions and :math:`H_{in} = \text{in\_features}`
        - Output: :math:`(N, *, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.
    """

    def __init__(self, in_features, out_features, initial_posterior_variance=1e-3, epsilon=1e-8):
        super().__init__(epsilon)
        self.in_features = in_features
        self.out_features = out_features
        self.ipv = initial_posterior_variance

        # priors are not optimizable parameters - all means and log-variances are zero
        self.register_buffer('prior_W_means', torch.zeros(out_features, in_features))
        self.register_buffer('prior_W_log_vars', torch.zeros(out_features, in_features))
        self.register_buffer('prior_b_means', torch.zeros(out_features))
        self.register_buffer('prior_b_log_vars', torch.zeros(out_features))

        self.posterior_W_means = Parameter(torch.empty_like(self._buffers['prior_W_means'], requires_grad=True))
        self.posterior_b_means = Parameter(torch.empty_like(self._buffers['prior_b_means'], requires_grad=True))
        self.posterior_W_log_vars = Parameter(torch.empty_like(self._buffers['prior_W_log_vars'], requires_grad=True))
        self.posterior_b_log_vars = Parameter(torch.empty_like(self._buffers['prior_b_log_vars'], requires_grad=True))

        self._initialize_posteriors()

    @weak_script_method
    def forward(self, x, sample_parameters=True):
        """ Produces module output on an input. """
        if sample_parameters:
            w, b = self._sample_parameters()
            return F.linear(x, w, b)
        else:
            return F.linear(x, self.posterior_W_means, self.posterior_b_means)

    def reset_for_next_task(self):
        """ Overwrites the current prior with the current posterior. """
        self._buffers['prior_W_means'].data.copy_(self.posterior_W_means.data)
        self._buffers['prior_W_log_vars'].data.copy_(self.posterior_W_log_vars.data)
        self._buffers['prior_b_means'].data.copy_(self.posterior_b_means.data)
        self._buffers['prior_b_log_vars'].data.copy_(self.posterior_b_log_vars.data)

    def kl_divergence(self) -> torch.Tensor:
        """ Returns KL(posterior, prior) for the parameters of this layer. """
        # obtain flattened means, log variances, and variances of the prior distribution
        prior_means = torch.autograd.Variable(torch.cat(
            (torch.reshape(self._buffers['prior_W_means'], (-1,)),
             torch.reshape(self._buffers['prior_b_means'], (-1,)))),
            requires_grad=False
        )
        prior_log_vars = torch.autograd.Variable(torch.cat(
            (torch.reshape(self._buffers['prior_W_log_vars'], (-1,)),
             torch.reshape(self._buffers['prior_b_log_vars'], (-1,)))),
            requires_grad=False
        )
        prior_vars = torch.exp(prior_log_vars)

        # obtain flattened means, log variances, and variances of the approximate posterior distribution
        posterior_means = torch.cat(
            (torch.reshape(self.posterior_W_means, (-1,)),
             torch.reshape(self.posterior_b_means, (-1,))),
        )
        posterior_log_vars = torch.cat(
            (torch.reshape(self.posterior_W_log_vars, (-1,)),
             torch.reshape(self.posterior_b_log_vars, (-1,))),
        )
        posterior_vars = torch.exp(posterior_log_vars)

        # compute kl divergence (this computation is valid for multivariate diagonal Gaussians)
        kl_elementwise = posterior_vars / (prior_vars + self.epsilon) + \
                         torch.pow(prior_means - posterior_means, 2) / (prior_vars + self.epsilon) - \
                         1 + prior_log_vars - posterior_log_vars

        return 0.5 * kl_elementwise.sum()

    def get_statistics(self) -> dict:
        statistics = {
            'average_w_mean': torch.mean(self.posterior_W_means),
            'average_b_mean': torch.mean(self.posterior_b_means),
            'average_w_var': torch.mean(torch.exp(self.posterior_W_log_vars)),
            'average_b_var': torch.mean(torch.exp(self.posterior_b_log_vars))
        }

        return statistics

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

    def _sample_parameters(self):
        # obtained sampled weights and biases using local reparameterization trick
        w_epsilons = torch.randn_like(self.posterior_W_means)
        b_epsilons = torch.randn_like(self.posterior_b_means)

        w = self.posterior_W_means + torch.mul(w_epsilons, torch.exp(0.5 * self.posterior_W_log_vars))
        b = self.posterior_b_means + torch.mul(b_epsilons, torch.exp(0.5 * self.posterior_b_log_vars))

        return w, b

    def _initialize_posteriors(self):
        # posteriors on the other hand are optimizable parameters - means are normally distributed, log_vars
        # have some small initial value
        torch.nn.init.normal_(self.posterior_W_means, mean=0, std=0.1)
        torch.nn.init.uniform_(self.posterior_b_means, -0.1, 0.1)
        torch.nn.init.constant_(self.posterior_W_log_vars, math.log(self.ipv))
        torch.nn.init.constant_(self.posterior_b_log_vars, math.log(self.ipv))
