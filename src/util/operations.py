import torch


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