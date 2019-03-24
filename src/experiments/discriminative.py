import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose
from models.vcl_nn import DiscriminativeVCL
from util.transforms import Flatten, Permute
from util.samplers import FilteringSampler
from util.outputs import write_as_json, save_model
from util.datasets import NOTMNIST
from tqdm import tqdm


# input and output dimensions of an FCFF MNIST classifier
MNIST_FLATTENED_DIM = 28 * 28
MNIST_N_CLASSES = 10
EPOCHS = 100
BATCH_SIZE = 256
LR = 0.001
# settings specific to permuted MNIST experiment
NUM_TASKS_PERM = 10
# settings specific to split MNIST experiment
LABEL_PAIRS_SPLIT = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]


def permuted_mnist():
    """
    Runs the 'Permuted MNIST' experiment from the VCL paper, in which each task
    is obtained by applying a fixed random permutation to the pixels of each image.
    """
    # flattening and permutation used for each task
    transforms = [Compose([Flatten(), Permute(torch.randperm(MNIST_FLATTENED_DIM))]) for _ in range(NUM_TASKS_PERM)]

    # create model, single-headed in permuted MNIST experiment
    model = DiscriminativeVCL(in_size=MNIST_FLATTENED_DIM, out_size=MNIST_N_CLASSES, layer_width=100, n_hidden_layers=2, n_tasks=1)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # each task is classification of MNIST images with permuted pixels
    for task in range(NUM_TASKS_PERM):
        print('TASK ' + str(task))

        mnist_train = MNIST(root='../data/', train=True, download=True, transform=transforms[task])
        train_loader = DataLoader(mnist_train, BATCH_SIZE)

        for _ in tqdm(range(EPOCHS), 'Epochs: '):
            for batch in train_loader:
                optimizer.zero_grad()
                x, y_true = batch

                # Note there is only one head on the model used for this.
                loss = model.loss(x, y_true, 0)
                loss.backward()
                optimizer.step()

    # test
    task_accuracies = []
    for task in tqdm(range(NUM_TASKS_PERM), 'Testing task: '):
        mnist_test = MNIST(root='../data/', train=False, download=True, transform=transforms[task])
        test_loader = DataLoader(mnist_test, batch_size=1)
        correct = 0

        for sample in test_loader:
            x, y_true = sample
            y_pred = torch.argmax(model.predict(x))

            if y_pred == y_true:
                correct += 1

        task_accuracies.append(correct / len(mnist_test))

    write_as_json('disc_p_mnist/accuracy.txt', task_accuracies)
    save_model(model, 'disc_p_mnist/model.pth')


def split_mnist():
    """
        Runs the 'Split MNIST' experiment from the VCL paper, in which each task
        is a binary classification task carried out on a subset of the MNIST dataset.
    """
    # download dataset
    mnist_train = MNIST(root='../data/', train=True, download=True, transform=Flatten())
    mnist_test = MNIST(root='../data/', train=False, download=True, transform=Flatten())

    # create model
    # fixme needs to be multi-headed
    # todo does it make sense to do binary classification with out_size=2 ?
    model = DiscriminativeVCL(in_size=MNIST_FLATTENED_DIM, out_size=2, layer_width=100, n_hidden_layers=2, n_tasks=5)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # each task is a binary classification task for a different pair of digits
    for task_idx, label_pair in enumerate(LABEL_PAIRS_SPLIT, 0):
        print('TASK ' + str(task_idx) + ', digits ' + str(label_pair))

        train_loader = DataLoader(mnist_train, BATCH_SIZE, sampler=FilteringSampler(mnist_train, label_pair))

        for _ in tqdm(range(EPOCHS), 'Epochs: '):
            for batch in train_loader:
                optimizer.zero_grad()
                x, y_true = batch

                # binarize labels - 1s where label is label_pair[1], 0 where it is label_pair[0]
                y_true = y_true == label_pair[1]

                loss = model.loss(x, y_true)
                loss.backward()
                optimizer.step()

    # test
    task_accuracies = []
    for task_idx, label_pair in enumerate(tqdm(LABEL_PAIRS_SPLIT, 'Testing task: '), 0):
        test_loader = DataLoader(mnist_test, batch_size=1, sampler=FilteringSampler(mnist_train, label_pair))
        correct = 0
        total = 0

        for sample_idx, sample in enumerate(test_loader, 1):
            # binarize labels - 1s where label is label_pair[1], 0 where it is label_pair[0]
            x, y_true = sample
            y_true = y_true == label_pair[1]

            y_pred = torch.round(model.prediction(x))

            if y_pred == y_true:
                correct += 1
            total = sample_idx

        task_accuracies.append(correct / total)

    write_as_json('disc_s_mnist/accuracy.txt', task_accuracies)
    save_model(model, 'disc_s_mnist/model.pth')


def split_not_mnist():
    """
        Runs the 'Split not MNIST' experiment from the VCL paper, in which each task
        is a binary classification task carried out on a subset of the not MNIST
        character recognition dataset.
    """
    not_mnist_train = NOTMNIST(train=True, overwrite=False, transform=Flatten(), limit_size=50000)
    not_mnist_test = NOTMNIST(train=False, overwrite=False, transform=Flatten())

    # create model
    # fixme needs to be multi-headed
    # todo does it make sense to do binary classification with out_size=2 ?
    model = DiscriminativeVCL(in_size=MNIST_FLATTENED_DIM, out_size=2, layer_width=100, n_hidden_layers=2)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # each task is a binary classification task for a different pair of characters
    for task_idx, label_pair in enumerate(LABEL_PAIRS_SPLIT, 0):
        print('TASK ' + str(task_idx) + ', chars (' + chr(label_pair[0] + 65) + ', ' + chr(label_pair[1] + 65) + ')')

        train_loader = DataLoader(not_mnist_train, BATCH_SIZE, sampler=FilteringSampler(not_mnist_train, label_pair))

        for _ in tqdm(range(EPOCHS), 'Epochs: '):
            for batch in train_loader:
                optimizer.zero_grad()
                x, y_true = batch

                # binarize labels - 1s where label is label_pair[1], 0 where it is label_pair[0]
                y_true = y_true == label_pair[1]

                loss = model.loss(x, y_true)
                loss.backward()
                optimizer.step()

    # test
    task_accuracies = []
    for task_idx, label_pair in enumerate(tqdm(LABEL_PAIRS_SPLIT, 'Testing task: '), 0):
        test_loader = DataLoader(not_mnist_test, batch_size=1, sampler=FilteringSampler(not_mnist_test, label_pair))
        correct = 0
        total = 0

        for sample_idx, sample in enumerate(test_loader, 1):
            # binarize labels - 1s where label is label_pair[1], 0 where it is label_pair[0]
            x, y_true = sample
            y_true = y_true == label_pair[1]

            y_pred = torch.round(model.prediction(x))

            if y_pred == y_true:
                correct += 1
            total = sample_idx

        task_accuracies.append(correct / total)

        write_as_json('disc_s_n_mnist/accuracy.txt', task_accuracies)
        save_model(model, 'disc_s_n_mnist/model.pth')
