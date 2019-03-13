import torch


def permute_dataset(dataset: torch.Tensor, permutation: torch.Tensor) -> torch.Tensor:
    """
    Applies the permutation specified by the given indices to each element of the dataset.

    Args:
        dataset: the dataset to permute, should be 2-dimensional tensor
        permutation: index array to apply to each image, should be 1-dimensional tensor

    Returns:
        The given dataset, after the provided permutation is applied to each element of
        the dataset.
    """
    return dataset[:, permutation]
