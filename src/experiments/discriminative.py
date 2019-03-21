import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose
from models.vcl_nn import VCL_NN
from util.transforms import Flatten, Permute
from util.outputs import write_as_json, save_model
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
    # flattening and permutation used for each task
    transforms = [Compose([Flatten(), Permute(torch.randperm(MNIST_FLATTENED_DIM))]) for _ in range(NUM_TASKS)]

    # create model
    model = VCL_NN(MNIST_FLATTENED_DIM, MNIST_N_CLASSES, 100, 2)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # each task is a random permutation of MNIST
    for task in range(NUM_TASKS):
        print('TASK ' + str(task))

        mnist_train = MNIST(root='../data/', train=True, download=True, transform=transforms[task])
        train_loader = DataLoader(mnist_train, BATCH_SIZE)

        for _ in tqdm(range(EPOCHS), 'Epochs: '):
            for batch in train_loader:
                optimizer.zero_grad()
                x, y_true = batch

                loss = model.loss(x, y_true)
                loss.backward()
                optimizer.step()

    # test
    task_accuracies = []
    for task in tqdm(range(NUM_TASKS), 'Testing task: '):
        mnist_test = MNIST(root='../data/', train=False, download=True, transform=transforms[task])
        test_loader = DataLoader(mnist_test, batch_size=1)
        correct = 0

        for sample in test_loader:
            x, y_true = sample
            y_pred = torch.argmax(model(x))

            if y_pred == y_true:
                correct += 1

        task_accuracies.append(correct / len(mnist_test))

    write_as_json('disc_p_mnist/accuracy.txt', task_accuracies)
    save_model(model, 'disc_p_mnist/model.pth')


def split_mnist():
    pass


def split_not_mnist():
    pass
