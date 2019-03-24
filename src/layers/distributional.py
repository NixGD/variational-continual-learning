import torch
import torch.nn.init
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch._jit_internal import weak_module, weak_script_method


@weak_module
class DistributionalLinear(torch.nn.Module):
    """
    A linear transformation on incoming data of the form :math:`y = w^T x + b`,
    where the weights w and the biases b are distributions of parameters
    rather than point estimates. This layer works in the same way as a standard
    fully connected layer, except that the parameters are Gaussian distributions
    rather than point estimates.

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

    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # priors are not optimizable parameters
        self.prior_W_means = torch.zeros(in_features, out_features)
        self.prior_W_log_vars = torch.zeros(in_features, out_features)
        self.prior_b_means = torch.zeros(out_features)
        self.prior_b_log_vars = torch.zeros(out_features)
        self.register_buffer('prior_W_means', self.prior_W_means)
        self.register_buffer('prior_W_log_vars', self.prior_W_log_vars)
        self.register_buffer('prior_b_means', self.prior_b_means)
        self.register_buffer('prior_b_log_vars', self.prior_b_log_vars)
        # posteriors on the other hand are optimizable parameters
        self.posterior_W_means = Parameter(self.prior_W_means.clone().detach(), requires_grad=True)
        self.posterior_W_log_vars = Parameter(self.prior_W_log_vars.clone().detach(), requires_grad=True)
        self.posterior_b_means = Parameter(self.prior_b_means.clone().detach(), requires_grad=True)
        self.posterior_b_log_vars = Parameter(self.prior_b_log_vars.clone().detach(), requires_grad=True)

    def reset_parameters(self):
        self.posterior_W_means = Parameter(self.prior_W_means.clone().detach(), requires_grad=True)
        self.posterior_W_log_vars = Parameter(self.prior_W_log_vars.clone().detach(), requires_grad=True)
        self.posterior_b_means = Parameter(self.prior_b_means.clone().detach(), requires_grad=True)
        self.posterior_b_log_vars = Parameter(self.prior_b_log_vars.clone().detach(), requires_grad=True)

    @weak_script_method
    def forward(self, input):
        W, b = self._sample_parameters()
        return F.linear(input, W, b)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

    def _sample_parameters(self):
        # obtained sampled weights and biases using local reparameterization trick
        W_epsilons = torch.randn_like(self.posterior_W_means)
        b_epsilons = torch.randn_like(self.posterior_b_means)

        sampled_W = self.posterior_W_means + torch.mul(W_epsilons, torch.exp(0.5 * self.posterior_W_log_vars))
        sampled_b = self.posterior_b_means + torch.mul(b_epsilons, torch.exp(0.5 * self.posterior_b_log_vars))

        return sampled_W, sampled_b
