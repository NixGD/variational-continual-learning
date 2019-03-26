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
    return 100 * (pred == true).sum().item() / len(true)


def concatenate_flattened(tensor_list) -> torch.Tensor:
    """
    Given list of tensors, flattens each and concatenates their values.
    """
    return torch.cat([torch.reshape(t, (-1,)) for t in tensor_list])


def task_subset(data: Dataset, task_ids: torch.Tensor, task: int,) -> torch.Tensor:
    idx_list = torch.arange(0, len(task_ids))[task_ids == task]
    return Subset(data, idx_list)
