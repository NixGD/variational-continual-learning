import os
from datetime import datetime
import torch
from torch.optim import Adam
from torchvision.transforms import Compose
from torchvision.datasets import MNIST
from models.contrib import GenerativeVCL
from models.coreset import RandomCoreset
from util.datasets import NOTMNIST
from util.transforms import Flatten, Scale
from util.experiment_utils import run_generative_point_estimate_initialisation, run_generative_task
from tensorboardX import SummaryWriter

MNIST_FLATTENED_DIM = 28 * 28
LR = 0.001
INITIAL_POSTERIOR_VAR = 1e-3

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Running on device", device)


def generate_mnist():
    """
        Runs the generative MNIST experiment from the VCL paper, in which each task is
        a generative task for one of the digits in the MNIST dataset.
    """
    z_dim = 50
    h_dim = 500
    layer_width = 500
    n_tasks = 10
    multiheaded = True
    coreset_size = 40
    epochs = 120
    batch_size = 50000

    transform = Compose([Flatten(), Scale()])

    # download dataset
    mnist_train = MNIST(root='data', train=True, download=True, transform=transform)
    mnist_test = MNIST(root='data/', train=False, download=True, transform=transform)

    model = GenerativeVCL(z_dim=z_dim, h_dim=h_dim, x_dim=MNIST_FLATTENED_DIM, n_heads=n_tasks,
                          encoder_h_dims=(layer_width, layer_width), decoder_head_h_dims=(layer_width,),
                          decoder_shared_h_dims=(layer_width,), initial_posterior_variance=INITIAL_POSTERIOR_VAR,
                          mc_sampling_n=10, device=device).to(device)

    optimizer = Adam(model.parameters(), lr=LR)
    coreset = RandomCoreset(size=coreset_size)

    # each label is its own task, so no need to define a dictionary like in the discriminative experiments
    if isinstance(mnist_train[0][1], int):
        train_task_ids = torch.Tensor([y for _, y in mnist_train])
        test_task_ids = torch.Tensor([y for _, y in mnist_test])
    elif isinstance(mnist_train[0][1], torch.Tensor):
        train_task_ids = torch.Tensor([y.item() for _, y in mnist_train])
        test_task_ids = torch.Tensor([y.item() for _, y in mnist_test])

    summary_logdir = os.path.join("logs", "disc_s_mnist", datetime.now().strftime('%b%d_%H-%M-%S'))
    writer = SummaryWriter(summary_logdir)

    # point estimate initialization is not done in the generative experiments
    # run_generative_point_estimate_initialisation(model=model, data=mnist_train,
    #                                              epochs=epochs, batch_size=batch_size,
    #                                              device=device, multiheaded=multiheaded,
    #                                              lr=LR, task_ids=train_task_ids,
    #                                              optimizer=optimizer)

    for task_idx in range(n_tasks):
        run_generative_task(
            model=model, train_data=mnist_train, train_task_ids=train_task_ids,
            test_data=mnist_test, test_task_ids=test_task_ids, coreset=coreset,
            task_idx=task_idx, epochs=epochs, batch_size=batch_size, lr=LR,
            save_as="disc_s_mnist", device=device, multiheaded=multiheaded,
            summary_writer=writer, optimizer=optimizer
        )

    writer.close()


def generate_not_mnist():
    """
        Runs the generative MNIST experiment from the VCL paper, in which each task is
        a generative task for one of the digits in the MNIST dataset.
    """
    z_dim = 50
    h_dim = 500
    layer_width = 500
    n_tasks = 10
    multiheaded = True
    coreset_size = 40
    epochs = 120
    batch_size = 50000

    transform = Compose([Flatten(), Scale()])

    # download dataset
    not_mnist_train = NOTMNIST(train=True, overwrite=False, transform=transform)
    not_mnist_test = NOTMNIST(train=False, overwrite=False, transform=transform)

    model = GenerativeVCL(z_dim=z_dim, h_dim=h_dim, x_dim=MNIST_FLATTENED_DIM, n_heads=n_tasks,
                          encoder_h_dims=(layer_width, layer_width), decoder_head_h_dims=(layer_width,),
                          decoder_shared_h_dims=(layer_width,), initial_posterior_variance=INITIAL_POSTERIOR_VAR,
                          mc_sampling_n=10, device=device).to(device)

    optimizer = Adam(model.parameters(), lr=LR)
    coreset = RandomCoreset(size=coreset_size)

    # each label is its own task, so no need to define a dictionary like in the discriminative experiments
    if isinstance(not_mnist_train[0][1], int):
        train_task_ids = torch.Tensor([y for _, y in not_mnist_train])
        test_task_ids = torch.Tensor([y for _, y in not_mnist_test])
    elif isinstance(not_mnist_train[0][1], torch.Tensor):
        train_task_ids = torch.Tensor([y.item() for _, y in not_mnist_train])
        test_task_ids = torch.Tensor([y.item() for _, y in not_mnist_test])

    summary_logdir = os.path.join("logs", "disc_s_mnist", datetime.now().strftime('%b%d_%H-%M-%S'))
    writer = SummaryWriter(summary_logdir)

    # point estimate initialization is not done in the generative experiments
    # run_generative_point_estimate_initialisation(model=model, data=not_mnist_train,
    #                                              epochs=epochs, batch_size=batch_size,
    #                                              device=device, multiheaded=multiheaded,
    #                                              lr=LR, task_ids=train_task_ids,
    #                                              optimizer=optimizer)

    for task_idx in range(n_tasks):
        run_generative_task(
            model=model, train_data=not_mnist_train, train_task_ids=train_task_ids,
            test_data=not_mnist_test, test_task_ids=test_task_ids, coreset=coreset,
            task_idx=task_idx, epochs=epochs, batch_size=batch_size, lr=LR,
            save_as="disc_s_mnist", device=device, multiheaded=multiheaded,
            summary_writer=writer, optimizer=optimizer
        )

    writer.close()
