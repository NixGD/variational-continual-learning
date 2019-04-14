import os
from datetime import datetime
import torch
import torch.nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Resize
from torchvision.datasets import MNIST
from models.contrib import GenerativeVCL
from models.coreset import RandomCoreset
from models.deep_models import MnistResNet
from util.datasets import NOTMNIST
from util.transforms import Flatten, Scale
from util.experiment_utils import run_generative_task
from util.operations import class_accuracy
from util.outputs import save_model, load_model
from tensorboardX import SummaryWriter
from tqdm import tqdm

MNIST_FLATTENED_DIM = 28 * 28
LR = 0.0001
INITIAL_POSTERIOR_VAR = 1e-12
CLASSIFIER_EPOCHS = 2
CLASSIFIER_BATCH_SIZE = 64
MNIST_CLASSIFIER_FILENAME = 'mnist_classifier.pth'
NOTMNIST_CLASSIFIER_FILENAME = 'n_mnist_classifier.pth'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Running on device", device)


def train_mnist_classifier():
    """
    Train a non-VCL classifier for MNIST to be used to compute the 'classifier uncertainty'
    evaluation metric in the generative tasks.
    """
    # image transforms and model
    model = MnistResNet().to(device)
    transforms = Compose([Resize(size=(224, 224)), ToTensor(), Scale()])
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adadelta(model.parameters())

    # download dataset
    mnist_train = MNIST(root="data", train=True, download=True, transform=transforms)
    mnist_test = MNIST(root="data", train=False, download=True, transform=transforms)
    train_loader = DataLoader(mnist_train, batch_size=CLASSIFIER_BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(mnist_test, batch_size=CLASSIFIER_BATCH_SIZE, shuffle=True)

    # train
    model.train()
    for epoch in tqdm(range(CLASSIFIER_EPOCHS), 'Epochs'):
        epoch_loss = 0
        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            x, y = batch[0].to(device), batch[1].to(device)

            predictions = model(x)
            loss = loss_fn(predictions, y)
            epoch_loss += len(x) * loss.item()

            loss.backward()
            optimizer.step()

    # evaluate
    model.eval()
    accuracies = []
    for batch in test_loader:
        x, y = batch[0].to(device), batch[1].to(device)

        predictions = torch.argmax(model(x), dim=1)
        accuracies.append(class_accuracy(predictions, y))

    accuracy = sum(accuracies) / len(accuracies)

    print('Classifier accuracy: ' + str(accuracy))
    save_model(model, MNIST_CLASSIFIER_FILENAME)


def train_not_mnist_classifier():
    """
    Train a non-VCL classifier for not-MNIST to be used to compute the 'classifier uncertainty'
    evaluation metric in the generative tasks.
    """
    # image transforms and model
    model = MnistResNet().to(device)
    transforms = Compose([Resize(size=(224, 224)), ToTensor(), Scale()])
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adadelta(model.parameters())

    # download dataset
    not_mnist_train = NOTMNIST(train=True, overwrite=False, transform=transforms)
    not_mnist_test = NOTMNIST(train=False, overwrite=False, transform=transforms)
    train_loader = DataLoader(not_mnist_train, CLASSIFIER_BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(not_mnist_test, CLASSIFIER_BATCH_SIZE, shuffle=True)

    # train
    model.train()
    for epoch in tqdm(range(CLASSIFIER_EPOCHS), 'Epochs'):
        epoch_loss = 0
        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            x, y = batch
            x = x.to(device)
            y = y.to(device)

            loss = loss_fn(model(x), y)
            epoch_loss += len(x) * loss.item()

            loss.backward()
            optimizer.step()

    # evaluate
    model.eval()
    accuracies = []
    for batch in test_loader:
        x, y = batch[0].to(device), batch[1].to(device)

        predictions = torch.argmax(model(x), dim=1)
        accuracies.append(class_accuracy(predictions, y))

    accuracy = sum(accuracies) / len(accuracies)

    print('Classifier accuracy: ' + str(accuracy))
    save_model(model, NOTMNIST_CLASSIFIER_FILENAME)


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
    epochs = 200
    batch_size = 50

    transform = Compose([Flatten(), Scale()])

    # download dataset
    mnist_train = MNIST(root='data', train=True, download=True, transform=transform)
    mnist_test = MNIST(root='data/', train=False, download=True, transform=transform)

    model = GenerativeVCL(z_dim=z_dim, h_dim=h_dim, x_dim=MNIST_FLATTENED_DIM, n_heads=n_tasks,
                          encoder_h_dims=(layer_width, layer_width, layer_width), decoder_head_h_dims=(layer_width,),
                          decoder_shared_h_dims=(layer_width,), initial_posterior_variance=INITIAL_POSTERIOR_VAR,
                          mc_sampling_n=10, device=device).to(device)
    evaluation_classifier = load_model(MNIST_CLASSIFIER_FILENAME).to(device)
    # we are using ResNet, so need to call eval()
    evaluation_classifier.eval()

    optimizer = Adam(model.parameters(), lr=LR)
    coreset = RandomCoreset(size=coreset_size)

    # each label is its own task, so no need to define a dictionary like in the discriminative experiments
    if isinstance(mnist_train[0][1], int):
        train_task_ids = torch.Tensor([y for _, y in mnist_train])
        test_task_ids = torch.Tensor([y for _, y in mnist_test])
    elif isinstance(mnist_train[0][1], torch.Tensor):
        train_task_ids = torch.Tensor([y.item() for _, y in mnist_train])
        test_task_ids = torch.Tensor([y.item() for _, y in mnist_test])

    summary_logdir = os.path.join("logs", "gen_mnist", datetime.now().strftime('%b%d_%H-%M-%S'))
    writer = SummaryWriter(summary_logdir)

    for task_idx in range(n_tasks):
        run_generative_task(
            model=model, train_data=mnist_train, train_task_ids=train_task_ids,
            test_data=mnist_test, test_task_ids=test_task_ids, coreset=coreset,
            task_idx=task_idx, epochs=epochs, batch_size=batch_size, lr=LR,
            save_as="gen_mnist", device=device, evaluation_classifier=evaluation_classifier,
            multiheaded=multiheaded, summary_writer=writer, optimizer=optimizer
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
    epochs = 400
    batch_size = 50

    transform = Compose([Flatten(), Scale()])

    # download dataset
    not_mnist_train = NOTMNIST(train=True, overwrite=False, transform=transform)
    not_mnist_test = NOTMNIST(train=False, overwrite=False, transform=transform)

    model = GenerativeVCL(z_dim=z_dim, h_dim=h_dim, x_dim=MNIST_FLATTENED_DIM, n_heads=n_tasks,
                          encoder_h_dims=(layer_width, layer_width, layer_width), decoder_head_h_dims=(layer_width,),
                          decoder_shared_h_dims=(layer_width,), initial_posterior_variance=INITIAL_POSTERIOR_VAR,
                          mc_sampling_n=10, device=device).to(device)
    evaluation_classifier = load_model(NOTMNIST_CLASSIFIER_FILENAME)
    # we are using ResNet, so need to call eval()
    evaluation_classifier.eval()

    optimizer = Adam(model.parameters(), lr=LR)
    coreset = RandomCoreset(size=coreset_size)

    # each label is its own task, so no need to define a dictionary like in the discriminative experiments
    if isinstance(not_mnist_train[0][1], int):
        train_task_ids = torch.Tensor([y for _, y in not_mnist_train])
        test_task_ids = torch.Tensor([y for _, y in not_mnist_test])
    elif isinstance(not_mnist_train[0][1], torch.Tensor):
        train_task_ids = torch.Tensor([y.item() for _, y in not_mnist_train])
        test_task_ids = torch.Tensor([y.item() for _, y in not_mnist_test])

    summary_logdir = os.path.join("logs", "gen_n_mnist", datetime.now().strftime('%b%d_%H-%M-%S'))
    writer = SummaryWriter(summary_logdir)

    for task_idx in range(n_tasks):
        run_generative_task(
            model=model, train_data=not_mnist_train, train_task_ids=train_task_ids,
            test_data=not_mnist_test, test_task_ids=test_task_ids, coreset=coreset,
            task_idx=task_idx, epochs=epochs, batch_size=batch_size, lr=LR,
            save_as="gen_n_mnist", device=device, evaluation_classifier=evaluation_classifier,
            multiheaded=multiheaded, summary_writer=writer, optimizer=optimizer
        )

    writer.close()
