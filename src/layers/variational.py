from abc import ABC, abstractmethod
import torch
import torch.nn.init
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch._jit_internal import weak_module, weak_script_method
import util.operations


@weak_module
class VariationalLayer(torch.nn.Module, ABC):
    @abstractmethod
    def forward(self, x, sample_parameters=True):
        pass

    @abstractmethod
    def reset_for_next_task(self):
        pass

    @abstractmethod
    def kl_divergence(self) -> float:
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

    __constants__ = ['bias']

    def __init__(self, in_features, out_features, initial_posterior_variance=1e-6, epsilon=1e-8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ipv = initial_posterior_variance
        self.epsilon = epsilon

        # priors are not optimizable parameters - all means and log-variances are zero
        self.register_buffer('prior_W_means', torch.zeros(out_features, in_features))
        self.register_buffer('prior_W_log_vars', torch.zeros(out_features, in_features))
        self.register_buffer('prior_b_means', torch.zeros(out_features))
        self.register_buffer('prior_b_log_vars', torch.zeros(out_features))

        self._initialize_posteriors()

    @weak_script_method
    def forward(self, x, sample_parameters=True):
        """ Produces module output on an input. """
        if sample_parameters:
            w, b = self._sample_parameters()
        else:
            w = self.posterior_W_means
            b = self.posterior_b_means

        return F.linear(x, w, b)

    def reset_for_next_task(self):
        """ Overwrites the current prior with the current posterior. """
        print('resetting')
        self._buffers['prior_W_means'] = self.posterior_W_means.clone().detach()
        self._buffers['prior_W_log_vars'] = self.posterior_W_log_vars.clone().detach()
        self._buffers['prior_b_means'] = self.posterior_b_means.clone().detach()
        self._buffers['prior_b_log_vars'] = self.posterior_b_log_vars.clone().detach()

    # def reset_parameters(self):
    #     """ Set the posteriors to the initial priors. """
    #     self._initialize_posteriors()

    def kl_divergence(self) -> float:
        """ Returns KL(posterior, prior) for the parameters of this layer. """
        # obtain flattened means, log variances, and variances of the prior distribution
        prior_means = util.operations.concatenate_flattened(self._buffers['prior_W_means'],
                                                            self._buffers['prior_b_means'])
        prior_log_vars = util.operations.concatenate_flattened(self._buffers['prior_W_log_vars'],
                                                               self._buffers['prior_b_log_vars'])
        prior_vars = torch.exp(prior_log_vars)

        # obtain flattened means, log variances, and variances of the approximate posterior distribution
        posterior_means = util.operations.concatenate_flattened(self.posterior_W_means, self.posterior_b_means)
        posterior_log_vars = util.operations.concatenate_flattened(self.posterior_W_log_vars, self.posterior_b_log_vars)
        posterior_vars = torch.exp(posterior_log_vars)

        # compute kl divergence (this computation is valid for multivariate diagonal Gaussians)
        kl_elementwise = posterior_vars / (prior_vars + self.epsilon) + \
                         torch.pow(prior_means - posterior_means, 2) / (prior_vars + self.epsilon) - \
                         1 + prior_log_vars - posterior_log_vars

        return 0.5 * kl_elementwise.sum().item()

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
        self.posterior_W_means = Parameter(torch.randn_like(self._buffers['prior_W_means'], requires_grad=True))
        self.posterior_b_means = Parameter(torch.randn_like(self._buffers['prior_b_means'], requires_grad=True))
        self.posterior_W_log_vars = Parameter(
            torch.full_like(self._buffers['prior_W_log_vars'], self.ipv, requires_grad=True))
        self.posterior_b_log_vars = Parameter(
            torch.full_like(self._buffers['prior_b_log_vars'], self.ipv, requires_grad=True))
