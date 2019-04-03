import torch
import torch.utils.data as data
from copy import deepcopy
import numpy as np
from util.operations import task_subset
import torch.optim as optim
from tqdm import tqdm

class Coreset():
    """
    Base class for the the coreset.  This version of the class has no
    coreset but subclasses will replace the select method.
    """

    def __init__(self, size=0, lr=0.001):
        self.size = size
        self.coreset = None
        self.coreset_task_ids = None
        self.lr = lr

    def select(self, d: data.Dataset, task_id: int):
        """
        Given a torch dataset, will choose k datapoints.  Will then update
        the coreset with these datapoints.
        Returns: the subset that was not selected as a torch dataset.
        """

        return d

    def coreset_train(self, m, old_optimizer, up_to_task, epochs, device,
                      y_transform=None, multiheaded=True, batch_size=256):
        """
        Returns a new model, trained on the coreset.  The returned model will
        be a deep copy, except when coreset is empty (when it will be identical)
        """

        if self.coreset is None:
            return m

        model = deepcopy(m)

        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        optimizer.load_state_dict(old_optimizer.state_dict())

        for task_idx in range(up_to_task+1):
            print('CORESET TASK', task_idx)

            head = task_idx if multiheaded else 0
            task_data = task_subset(self.coreset, self.coreset_task_ids, task_idx)
            train_loader = data.DataLoader(task_data, batch_size)
            for _ in tqdm(range(epochs), 'Epochs: '):
                for batch in train_loader:
                    optimizer.zero_grad()
                    x, y_true = batch
                    x = x.to(device)
                    y_true = y_true.to(device)

                    if y_transform is not None:
                        y_true = y_transform(y_true, task_idx)

                    loss = model.vcl_loss(x, y_true, len(task_data), head)
                    loss.backward()
                    optimizer.step()

        return model


class RandomCoreset(Coreset):

    def __init__(self, size):
        super().__init__(size)

    def select(self, d : data.Dataset, task_id : int):

        new_cs_data, non_cs = data.random_split(d, [self.size, max(0,len(d)-self.size)])

        # Need to split the x from the y values to also include the task values.
        # I don't like this way of doing it, but I couldn't find something better.
        new_cs_x = torch.tensor([x for x, _ in new_cs_data])
        new_cs_y = torch.tensor([y for _, y in new_cs_data])

        new_cs = data.TensorDataset(new_cs_x, new_cs_y)
        new_task_ids = torch.full((len(new_cs_data),), task_id)

        if self.coreset is None:
            self.coreset = new_cs
            self.coreset_task_ids = new_task_ids
        else:
            self.coreset = data.ConcatDataset((self.coreset, new_cs))
            self.coreset_task_ids = torch.cat((self.coreset_task_ids, new_task_ids))

        return non_cs
