import torch
from torch.utils.data import Dataset, DataLoader
from copy import deepcopy

class Coreset():
    """
    Base class for the the coreset.  This version of the class has no
    coreset but subclasses will replace the select method.
    """

    def __init__(self, size=0):
        self.size = size
        self.coreset = data.DataSet()

    def select(self, data: Dataset, task_id: int):
        """
        Given a torch dataset, will choose k datapoints.  Will then update
        the coreset with these datapoints.
        Returns: the subset that was not selected as a torch dataset.
        """

        return data

    def coreset_train(self, m, optimizer, batch_size = 256):
        """
        Returns a new model, trained on the coreset.  The returned model will
        be a deep copy, except when coreset is empty (when it will be identical)
        """

        if not len(self.coreset):
            return m

        model = deepcopy(m)

        for batch in DataLoader(self.coreset, batch_size):
            optimizer.zero_grad()
            x, y_true, task = batch

            loss = model.loss(x, y_true, task)
            loss.backward()
            optimizer.step()

        return model
