import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from models.vcl_nn_reworked import DiscriminativeVCL, GenerativeVCL
from util.transforms import Flatten
from util.outputs import write_as_json, save_model
from tqdm import tqdm

N_TASKS = 10
MNIST_FLATTENED_DIM = 28 * 28
Z_DIM = 50
H_DIM = 500
HIDDEN_WIDTH = 500
EPOCHS = 100
BATCH_SIZE = 256
LR = 0.001


def generate_mnist():
    """
            Runs the 'Split MNIST' experiment from the VCL paper, in which each task
            is a binary classification task carried out on a subset of the MNIST dataset.
        """
    # download dataset
    mnist_train = MNIST(root='../data/', train=True, download=True, transform=Flatten())
    mnist_test = MNIST(root='../data/', train=False, download=True, transform=Flatten())

    # create multi-headed VCL decoder (task-specific encoders created later)
    decoder = GenerativeVCL(z_dim=Z_DIM, h_dim=H_DIM, x_dim=MNIST_FLATTENED_DIM, n_heads=10, n_hidden_layers=(1, 1),
                            hidden_dims=(500, 500))
    optimizer = Adam(decoder.parameters(), lr=LR)

    # each task is a generative task for one specific digit
    for task in range(N_TASKS):
        print('TASK ' + str(task) + ':')
        train_loader = DataLoader(mnist_train, BATCH_SIZE)

        # every task has a task-specific encoder, symmetric to the decoder todo make symmetric
        encoder = DiscriminativeVCL(x_dim=MNIST_FLATTENED_DIM, h_dim=Z_DIM, layer_width=HIDDEN_WIDTH, n_hidden_layers=3)
        optimizer.add_param_group(encoder.parameters())

        for _ in tqdm(range(EPOCHS), 'Epochs: '):
            for batch in train_loader:
                optimizer.zero_grad()
                x, _ = batch

                z = encoder(x)
                x = decoder(z)

                loss = decoder.loss(x, encoder)
                loss.backward()
                optimizer.step()

        optimizer.param_groups.remove(-1)
    # test
    task_accuracies = []
    for task in tqdm(range(N_TASKS), 'Testing task: '):
        test_loader = DataLoader(mnist_test, batch_size=1)
        correct = 0
        total = 0

        for sample_idx, sample in enumerate(test_loader, 1):
            # binarize labels - 1s where label is label_pair[1], 0 where it is label_pair[0]
            x, _ = sample
            # todo

        task_accuracies.append(correct / total)

    write_as_json('disc_s_mnist/accuracy.txt', task_accuracies)
    save_model(model, 'disc_s_mnist/model.pth')


def generate_not_mnist():
    pass
