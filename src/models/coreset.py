import torch
import torch.utils.data as data
from copy import deepcopy
import numpy as np

class Coreset():
    """
    Base class for the the coreset.  This version of the class has no
    coreset but subclasses will replace the select method.
    """

    def __init__(self, size=0):
        self.size = size
        self.coreset = None
        self.coreset_task_ids = None

    def select(self, d: data.Dataset, task_id: int):
        """
        Given a torch dataset, will choose k datapoints.  Will then update
        the coreset with these datapoints.
        Returns: the subset that was not selected as a torch dataset.
        """

        return d

    def coreset_train(self, m, optimizer, batch_size = 256):
        """
        Returns a new model, trained on the coreset.  The returned model will
        be a deep copy, except when coreset is empty (when it will be identical)
        """

        if self.coreset is not None:
            return m

        model = deepcopy(m)

        for batch in data.DataLoader(self.coreset, batch_size):
            optimizer.zero_grad()
            x, y_true, task = batch

            loss = model.loss(x, y_true, task)
            loss.backward()
            optimizer.step()

        return model


class RandomCoreset(Coreset):

    def __init__(self, size):
        super().__init__(size)

    def select(self, d : data.Dataset, task_id : int):

        new_cs_data, non_cs = data.random_split(d, [self.size, max(0,len(d)-self.size)])

        new_cs_x = torch.tensor([x for x,y in new_cs_data])
        new_cs_y = torch.tensor([y for x,y in new_cs_data])

        new_cs = data.TensorDataset(
                        new_cs_x, new_cs_y,
                        torch.full((len(new_cs_data),), task_id)
                        )

        if self.coreset is None:
            self.corset = new_cs
        else:
            self.coreset = data.ConcatDataset((self.corset, new_cs))

        return non_cs
