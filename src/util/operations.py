import torch
from torch.utils.data import Dataset, Subset


def class_accuracy(pred: torch.Tensor, true: torch.Tensor) -> float:
    """
    Computes the percentage class accuracy of the predictions, given the correct
    class labels.

    Args:
        pred: the class predictions made by a model
        true: the ground truth classes of the sample
    Returns:
        Classification accuracy of the predictions w.r.t. the ground truth labels
    """
    return 100 * (pred.int() == true.int()).sum().item() / len(true)


def kl_divergence(z_posterior_means, z_posterior_log_std, z_prior_mean=0.0, z_prior_log_std=0.0):
    """ Computes KL(z_posterior, z_prior) """
    # code adapted from author implementation at
    # https://github.com/nvcuong/variational-continual-learning/blob/master/dgm/alg/helper_functions.py
    z_prior_means = torch.full_like(z_posterior_means, z_prior_mean)
    z_prior_log_stds = torch.full_like(z_posterior_log_std, z_prior_log_std)

    prior_precision = torch.exp(torch.mul(z_prior_log_stds, -2))
    kl = 0.5 * ((z_posterior_means - z_prior_means) ** 2) * prior_precision - 0.5
    kl += z_prior_log_stds - z_posterior_log_std
    kl += 0.5 * torch.exp(2 * z_posterior_log_std - 2 * z_prior_log_stds)
    return torch.sum(kl, dim=(1,))


def bernoulli_log_likelihood(x_observed, x_reconstructed, epsilon=1e-8) -> torch.Tensor:
    """
    For observed batch of data x, and reconstructed data p (we view p as a
    probability of a pixel being on), computes a tensor of dimensions
    [batch_size] representing the log likelihood of each data point in the batch.
    """
    # broken into steps because some log probabilities are extremely low and cause NaNs to appear
    # as a hacky solution, we replace NaNs with a log probability of 10**-8 as an intermediate step
    prob = torch.mul(torch.log(x_reconstructed + epsilon), x_observed)
    inv_prob = torch.mul(torch.log(1 - x_reconstructed + epsilon), 1 - x_observed)
    inv_prob[inv_prob != inv_prob] = epsilon

    return torch.sum(torch.add(prob, inv_prob), dim=(1,))


def normal_with_reparameterization(means: torch.Tensor, log_stds: torch.Tensor, device='cpu') -> torch.Tensor:
    return torch.add(means, torch.mul(torch.exp(log_stds), torch.randn_like(means)))


def concatenate_flattened(tensor_list) -> torch.Tensor:
    """
    Given list of tensors, flattens each and concatenates their values.
    """
    return torch.cat([torch.reshape(t, (-1,)) for t in tensor_list])


def task_subset(data: Dataset, task_ids: torch.Tensor, task: int,) -> torch.Tensor:
    idx_list = torch.arange(0, len(task_ids))[task_ids == task]
    return Subset(data, idx_list)
