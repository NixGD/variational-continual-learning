import numpy as np
import torch
import torch.optim as optim
import torchvision.datasets as datasets
from models.vcl_nn import VCL_NN
from util.dataset_transforms import permute_dataset
from util.operations import class_accuracy
from util.outputs import write_as_json
from tqdm import tqdm


# input and output dimensions of an FCFF MNIST classifier
MNIST_FLATTENED_DIM = 28 * 28
MNIST_N_CLASSES = 10
EPOCHS = 100
BATCH_SIZE = 256
LR = 0.001
NUM_TASKS = 10


def permuted_mnist():
    """
    Runs the 'Permuted MNIST' experiment from the VCL paper, in which each task
    is obtained by applying a fixed random permutation to the pixels of each image.
    """
    # download MNIST and convert entire dataset to tensors for easy permutations
    mnist_train = datasets.MNIST(root='./data/', train=True, download=True)
    mnist_test = datasets.MNIST(root='./data/', train=False, download=True)

    # create tensors for train dataset
    x_train = torch.tensor([np.array(image[0]) for image in mnist_train])
    x_train = torch.reshape(x_train, shape=(len(x_train), MNIST_FLATTENED_DIM))
    y_train = torch.tensor([image[1] for image in mnist_train])
    # create tensors for test dataset
    x_test = torch.tensor([np.array(image[0]) for image in mnist_test])
    x_test = torch.reshape(x_test, shape=(len(x_test), MNIST_FLATTENED_DIM))
    y_test = torch.tensor([image[1] for image in mnist_test])

    # pixel permutation used for each task
    perms = [torch.randperm(MNIST_FLATTENED_DIM) for _ in range(NUM_TASKS)]

    # create model
    model = VCL_NN(MNIST_FLATTENED_DIM, MNIST_N_CLASSES, 100, 2)
    # optimizer = optim.Adam(model.parameters(), lr=LR)

    # each task is a random permutation of MNIST
    for task in tqdm(range(NUM_TASKS), 'Training task: '):
        x_train = permute_dataset(x_train, perms[task])

        for e in range(EPOCHS):
            # todo optimization code - how to implement not yet decided
            pass

    # test
    accuracy = []
    for task in tqdm(range(NUM_TASKS), 'Testing task: '):
        x_test = permute_dataset(x_test, perms[task])
        # pred = model(x_test)
        # accuracy.append(class_accuracy(pred, y_test))
    write_as_json('disc_p_mnist/accuracy.txt', accuracy)


def split_mnist():
    pass


def split_not_mnist():
    pass
