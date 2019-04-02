import torch
import torch.optim as optim
import numpy as np
from torchvision.datasets import MNIST
from torchvision.transforms import Compose
from torch.utils.data import ConcatDataset
from models.vcl_nn import DiscriminativeVCL
from models.coreset import RandomCoreset
from util.experiment_utils import run_point_estimate_initialisation, run_task
from util.transforms import Flatten, Permute
from util.datasets import NOTMNIST
from tensorboardX import SummaryWriter
import os
from datetime import datetime

MNIST_FLATTENED_DIM = 28 * 28
LR = 0.001
INITIAL_POSTERIOR_VAR = 1e-2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Running on device", device)

def permuted_mnist():
    """
    Runs the 'Permuted MNIST' experiment from the VCL paper, in which each task
    is obtained by applying a fixed random permutation to the pixels of each image.
    """
    N_CLASSES = 10
    LAYER_WIDTH = 100
    N_HIDDEN_LAYERS = 2
    N_TASKS = 10
    CORESET_SIZE = 200
    EPOCHS = 100
    BATCH_SIZE = 256

    # flattening and permutation used for each task
    transforms = [Compose([Flatten(), Permute(torch.randperm(MNIST_FLATTENED_DIM))]) for _ in range(N_TASKS)]

    # create model, single-headed in permuted MNIST experiment
    model = DiscriminativeVCL(
        in_size=MNIST_FLATTENED_DIM, out_size=N_CLASSES,
        layer_width=LAYER_WIDTH, n_hidden_layers=N_HIDDEN_LAYERS,
        n_tasks=N_TASKS, initial_posterior_var=INITIAL_POSTERIOR_VAR
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    coreset = RandomCoreset(size=CORESET_SIZE)

    mnist_train = ConcatDataset(
        [MNIST(root='../data/', train=True, download=True, transform=t) for t in transforms]
    )
    task_size = len(mnist_train) // N_TASKS
    train_task_ids = torch.cat(
        [torch.full((task_size,), id) for id in range(N_TASKS)]
    )

    mnist_test = ConcatDataset(
        [MNIST(root='../data/', train=False, download=True, transform=t) for t in transforms]
    )
    task_size = len(mnist_test) // N_TASKS
    test_task_ids = torch.cat(
        [torch.full((task_size,), id) for id in range(N_TASKS)]
    )

    summary_logdir = os.path.join("logs", "disc_p_mnist", datetime.now().strftime('%b%d_%H-%M-%S'))
    writer = SummaryWriter(summary_logdir)
    run_point_estimate_initialisation(model=model, data=mnist_train,
                                      optimizer=optimizer, epochs=EPOCHS,
                                      batch_size=BATCH_SIZE, device=device,
                                      task_ids=train_task_ids)

    # each task is classification of MNIST images with permuted pixels
    for task in range(N_TASKS):
        run_task(
            model=model, train_data=mnist_train, train_task_ids=train_task_ids,
            test_data=mnist_test, test_task_ids=test_task_ids, task_idx=task,
            coreset=coreset, optimizer=optimizer, epochs=EPOCHS,
            batch_size=BATCH_SIZE, device=device, save_as="disc_p_mnist",
            multiheaded=False, summary_writer=writer
        )
        model.reset_for_new_task(task)

    writer.close()

def split_mnist():
    """
    Runs the 'Split MNIST' experiment from the VCL paper, in which each task is
    a binary classification task carried out on a subset of the MNIST dataset.
    """
    N_CLASSES = 2 # TODO does it make sense to do binary classification with out_size=2 ?
    LAYER_WIDTH = 256
    N_HIDDEN_LAYERS = 2
    N_TASKS = 5
    CORESET_SIZE = 40
    EPOCHS = 120
    BATCH_SIZE = 50000

    # download dataset
    mnist_train = MNIST(root='../data/', train=True, download=True, transform=Flatten())
    mnist_test = MNIST(root='../data/', train=False, download=True, transform=Flatten())

    model = DiscriminativeVCL(
        in_size=MNIST_FLATTENED_DIM, out_size=N_CLASSES,
        layer_width=LAYER_WIDTH, n_hidden_layers=N_HIDDEN_LAYERS,
        n_tasks=N_TASKS, initial_posterior_var=INITIAL_POSTERIOR_VAR
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    coreset = RandomCoreset(size=CORESET_SIZE)

    label_to_task_mapping = {
        0: 0, 1: 0,
        2: 1, 3: 1,
        4: 2, 5: 2,
        6: 3, 7: 3,
        8: 4, 9: 4,
    }

    if isinstance(mnist_train[0][1], int):
        train_task_ids = torch.Tensor([label_to_task_mapping[y] for _, y in mnist_train])
        test_task_ids = torch.Tensor([label_to_task_mapping[y] for _, y in mnist_test])
    elif isinstance(mnist_train[0][1], torch.Tensor):
        train_task_ids = torch.Tensor([label_to_task_mapping[y.item()] for _, y in mnist_train])
        test_task_ids = torch.Tensor([label_to_task_mapping[y.item()] for _, y in mnist_test])

    summary_logdir = os.path.join("logs", "disc_s_mnist", datetime.now().strftime('%b%d_%H-%M-%S'))
    writer = SummaryWriter(summary_logdir)

    # each task is a binary classification task for a different pair of digits
    binarize_y = lambda y, task: (y == (2 * task + 1)).long()

    run_point_estimate_initialisation(model=model, data=mnist_train,
                                      optimizer=optimizer, epochs=EPOCHS,
                                      batch_size=BATCH_SIZE, device=device,
                                      task_ids=train_task_ids, y_transform=binarize_y)

    for task_idx in range(N_TASKS):
        run_task(
            model=model, train_data=mnist_train, train_task_ids=train_task_ids,
            test_data=mnist_test, test_task_ids=test_task_ids, optimizer=optimizer,
            coreset=coreset, task_idx=task_idx, epochs=EPOCHS, batch_size=BATCH_SIZE,
            save_as="disc_s_mnist", device=device, y_transform=binarize_y,
            summary_writer=writer
        )
        model.reset_for_new_task(task_idx)

    writer.close()

def split_not_mnist():
    """
        Runs the 'Split not MNIST' experiment from the VCL paper, in which each task
        is a binary classification task carried out on a subset of the not MNIST
        character recognition dataset.
    """
    N_CLASSES = 2 # TODO does it make sense to do binary classification with out_size=2 ?
    LAYER_WIDTH = 150
    N_HIDDEN_LAYERS = 4
    N_TASKS = 5
    CORESET_SIZE = 40
    EPOCHS = 120
    BATCH_SIZE = 400000

    not_mnist_train = NOTMNIST(train=True, overwrite=False, transform=Flatten())
    not_mnist_test = NOTMNIST(train=False, overwrite=False, transform=Flatten())

    model = DiscriminativeVCL(
        in_size=MNIST_FLATTENED_DIM, out_size=N_CLASSES,
        layer_width=LAYER_WIDTH, n_hidden_layers=N_HIDDEN_LAYERS,
        n_tasks=N_TASKS, initial_posterior_var=INITIAL_POSTERIOR_VAR
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    coreset = RandomCoreset(size=CORESET_SIZE)

    # todo: are the y classes integers?  Or characters?
    label_to_task_mapping = {
        0: 0, 1: 0,
        2: 1, 3: 1,
        4: 2, 5: 2,
        6: 3, 7: 3,
        8: 4, 9: 4,
    }

    train_task_ids = torch.from_numpy(np.array([label_to_task_mapping[y] for _, y in not_mnist_train]))
    test_task_ids = torch.from_numpy(np.array([label_to_task_mapping[y] for _, y in not_mnist_test]))

    summary_logdir = os.path.join("logs", "disc_s_n_mnist", datetime.now().strftime('%b%d_%H-%M-%S'))
    writer = SummaryWriter(summary_logdir)

    # each task is a binary classification task for a different pair of digits
    binarize_y = lambda y, task: (y == (2 * task + 1)).long()

    run_point_estimate_initialisation(model=model, data=not_mnist_train,
                                      optimizer=optimizer, epochs=EPOCHS,
                                      batch_size=BATCH_SIZE, device=device,
                                      task_ids=train_task_ids, y_transform=binarize_y)

    for task_idx in range(N_TASKS):
        run_task(
            model=model, train_data=not_mnist_train, train_task_ids=train_task_ids,
            test_data=not_mnist_test, test_task_ids=test_task_ids,
            optimizer=optimizer, coreset=coreset, task_idx=task_idx, epochs=EPOCHS,
            batch_size=BATCH_SIZE, save_as="disc_s_n_mnist", device=device,
            y_transform=binarize_y, summary_writer=writer
        )
        model.reset_for_new_task(task_idx)

    writer.close()
