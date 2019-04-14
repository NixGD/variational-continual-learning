import numpy as np


class Flatten(object):
    """ Transforms a PIL image to a flat numpy array. """
    def __init__(self):
        pass

    def __call__(self, sample):
        return np.array(sample, dtype=np.float32).flatten()


class Scale(object):
    """Scale images down to have [0,1] float pixel values"""
    def __init__(self, max_value=255):
        self.max_value = max_value

    def __call__(self, sample):
        return sample / self.max_value


class Permute(object):
    """ Apply a fixed permutation to the pixels in the image. """
    def __init__(self, permutation):
        self.permutation = permutation

    def __call__(self, sample):
        return sample[self.permutation]
