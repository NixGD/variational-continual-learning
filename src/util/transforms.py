import numpy as np


class Flatten(object):
    """ Transforms a PIL image to a flat numpy array. """
    def __init__(self):
        pass

    def __call__(self, sample):
        return np.array(sample, dtype=np.float32).flatten()


class Permute(object):
    """ Apply a fixed permutation to the pixels in the image. """
    def __init__(self, permutation):
        self.permutation = permutation

    def __call__(self, sample):
        return sample[self.permutation]
