from torch.utils.data.sampler import Sampler


class FilteringSampler(Sampler):
    """
    A Sampler which returns sequentially sampled elements from the given data source,
    provided that their class label is part of a list of desired labels. That is, allows
    sampling exclusively from a desired subset of classes.
    """
    def __init__(self, data_source, labels):
        super().__init__(data_source)
        self.data_source = data_source
        self.labels = []
        self.labels.extend(labels)

        unfiltered_indices = [idx if img[1] in self.labels else None for idx, img in enumerate(self.data_source, 0)]
        self.indices = list(filter(lambda idx: idx is not None, unfiltered_indices))

    def __iter__(self):
        # must return a list of indices
        return iter(self.indices)

    def __len__(self):
        # not really needed, implement if ever required
        raise NotImplementedError()
