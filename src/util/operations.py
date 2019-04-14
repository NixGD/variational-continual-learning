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


def kl_divergence(posterior_means, posterior_log_vars, prior_mean=0.0, prior_log_var=0.0):
    """ Computes KL(posterior, prior) """
    # code adapted from author implementation at
    # https://github.com/nvcuong/variational-continual-learning/blob/master/dgm/alg/helper_functions.py
    p_means = torch.full_like(posterior_means, prior_mean)
    p_log_vars = torch.full_like(posterior_log_vars, prior_log_var)
    prior_precision = torch.exp(torch.mul(p_log_vars, -2))
    kl = 0.5 * (posterior_means - p_means) ** 2 * prior_precision - 0.5
    kl += p_log_vars - posterior_log_vars
    kl += 0.5 * torch.exp(2 * posterior_log_vars - 2 * p_log_vars)
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

    return torch.sum(torch.add(prob, inv_prob), 1)


def concatenate_flattened(tensor_list) -> torch.Tensor:
    """
    Given list of tensors, flattens each and concatenates their values.
    """
    return torch.cat([torch.reshape(t, (-1,)) for t in tensor_list])


def task_subset(data: Dataset, task_ids: torch.Tensor, task: int,) -> torch.Tensor:
    idx_list = torch.arange(0, len(task_ids))[task_ids == task]
    return Subset(data, idx_list)
