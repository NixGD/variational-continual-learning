"""
Utilities that abstract the low-level details of experiments, such as standard train-and-eval loops.
"""

import torch
import torch.optim as optim
import torch.nn.functional as F
from util.operations import task_subset, class_accuracy, bernoulli_log_likelihood
from util.outputs import write_as_json, save_model
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from util.plot_autograd import make_dot


def run_point_estimate_initialisation(model, data, epochs, task_ids, batch_size,
                                      device, lr, task_idx=0, y_transform=None,
                                      multiheaded=True):
    """
    Trains a VCL model in 'non-variational' mode on a task to learn good initial estimates
    for the model parameter means.

    :param model: the VCL model to train
    :param data: the complete dataset to train on, such as MNIST
    :param epochs: number of training epochs
    :param task_ids: the label-to-task mapping that defines which task examples in the dataset belong to
    :param batch_size: batch size used in training
    :param device: device to run the experiment on, either 'cpu' or 'cuda'
    :param lr: optimizer learning rate to use
    :param task_idx: task being learned, maps to a specific head in the network
    :param y_transform: transform to be applied to the dataset labels
    :param multiheaded: true if the network being trained is multi-headed
    """
    print("Obtaining point estimate for posterior initialisation")

    head = task_idx if multiheaded else 0

    # each task has its own optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # obtain the appropriate data subset depending on which task we are running
    task_data = task_subset(data, task_ids, task_idx)
    loader = DataLoader(task_data, batch_size)

    # train
    for _ in tqdm(range(epochs), 'Epochs: '):
        for batch in loader:
            optimizer.zero_grad()
            x, y_true = batch
            x = x.to(device)
            y_true = y_true.to(device)

            if y_transform is not None:
                y_true = y_transform(y_true, task_idx)

            loss = model.point_estimate_loss(x, y_true, head=head)
            loss.backward()
            optimizer.step()


def run_task(model, train_data, train_task_ids, test_data, test_task_ids,
             task_idx, coreset, epochs, batch_size, save_as, device, lr,
             y_transform=None, multiheaded=True, train_full_coreset=True,
             summary_writer=None):
    """
        Trains a VCL model using online variational inference on a task, and performs a coreset
        training run as well as an evaluation after training.

        :param model: the VCL model to train
        :param train_data: the complete dataset to train on, such as MNIST
        :param train_task_ids: the label-to-task mapping that defines which task examples in the dataset belong to
        :param test_data: the complete dataset to train on, such as MNIST
        :param test_task_ids: the label-to-task mapping that defines which task examples in the dataset belong to
        :param task_idx: task being learned, maps to a specific head in the network
        :param coreset: coreset object to use in training
        :param epochs: number of training epochs
        :param batch_size: batch size used in training
        :param save_as: base directory to save into
        :param device: device to run the experiment on, either 'cpu' or 'cuda'
        :param lr: optimizer learning rate to use
        :param y_transform: transform to be applied to the dataset labels
        :param multiheaded: true if the network being trained is multi-headed
        :param summary_writer: tensorboard_x summary writer
        """

    print('TASK ', task_idx)

    # separate optimizer for each task
    optimizer = optim.Adam(model.parameters(), lr=lr)

    head = task_idx if multiheaded else 0

    # obtain correct subset of data for training, and set some aside for the coreset
    task_data = task_subset(train_data, train_task_ids, task_idx)
    non_coreset_data = coreset.select(task_data, task_id=task_idx)
    train_loader = DataLoader(non_coreset_data, batch_size)

    # train
    for epoch in tqdm(range(epochs), 'Epochs: '):
        epoch_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            x, y_true = batch
            x = x.to(device)
            y_true = y_true.to(device)

            if y_transform is not None:
                y_true = y_transform(y_true, task_idx)

            loss = model.vcl_loss(x, y_true, head, len(task_data))
            epoch_loss += len(x) * loss.item()

            loss.backward()
            optimizer.step()

        if summary_writer is not None:
            summary_writer.add_scalars("loss", {"TASK_" + str(task_idx): epoch_loss / len(task_data)}, epoch)

    # after training, prepare for new task by copying posteriors into priors
    model.reset_for_new_task(head)

    # train using full coreset
    if train_full_coreset:
        model_cs_trained = coreset.coreset_train(
            model, optimizer, list(range(task_idx+1)), epochs,
            device, y_transform=y_transform, multiheaded=multiheaded )

    # test
    task_accuracies = []
    tot_right = 0
    tot_tested = 0

    for test_task_idx in range(task_idx + 1):
        if not train_full_coreset:
            model_cs_trained = coreset.coreset_train(
                model, optimizer, test_task_idx , epochs,
                device, y_transform=y_transform, multiheaded=multiheaded )

        head = test_task_idx if multiheaded else 0

        task_data = task_subset(test_data, test_task_ids, test_task_idx)

        x = torch.Tensor([x for x, _ in task_data])
        y_true = torch.Tensor([y for _, y in task_data])
        x = x.to(device)
        y_true = y_true.to(device)

        if y_transform is not None:
            y_true = y_transform(y_true, test_task_idx)

        y_pred = model_cs_trained.prediction(x, head)

        acc = class_accuracy(y_pred, y_true)
        print("After task {} perfomance on task {} is {}"
              .format(task_idx, test_task_idx, acc))

        tot_right += acc * len(task_data)
        tot_tested += len(task_data)
        task_accuracies.append(acc)

    mean_accuracy = tot_right / tot_tested
    print("Mean accuracy:", mean_accuracy)

    if summary_writer is not None:
        task_accuracies_dict = dict(zip(["TASK_" + str(i) for i in range(task_idx + 1)], task_accuracies))
        summary_writer.add_scalars("test_accuracy", task_accuracies_dict, task_idx + 1)
        summary_writer.add_scalar("mean_posterior_variance", model._mean_posterior_variance(), task_idx + 1)
        summary_writer.add_scalar("mean_accuracy", mean_accuracy, task_idx + 1)

    write_as_json(save_as + '/accuracy.txt', task_accuracies)
    save_model(model, save_as + '_model_task_' + str(task_idx) + '.pth')


def run_generative_point_estimate_initialisation(model, data, epochs, task_ids, batch_size,
                                                 device, lr, task_idx=0,
                                                 multiheaded=True, optimizer=None):
    """
    Trains a generative VCL model in 'non-variational' mode on a task to learn good initial estimates
    for the model parameter means.

    :param model: the VCL model to train
    :param data: the complete dataset to train on, such as MNIST
    :param epochs: number of training epochs
    :param task_ids: the label-to-task mapping that defines which task examples in the dataset belong to
    :param batch_size: batch size used in training
    :param device: device to run the experiment on, either 'cpu' or 'cuda'
    :param lr: optimizer learning rate to use
    :param task_idx: task being learned, maps to a specific head in the network
    :param multiheaded: true if the network being trained is multi-headed
    :param optimizer: optionally, provide an existing optimizer instead of having the method create a new one
    """
    print("Obtaining point estimate for posterior initialisation")

    head = task_idx if multiheaded else 0

    # each task has its own optimizer
    optimizer = optimizer if optimizer is not None else optim.Adam(model.parameters(), lr=lr)

    # obtain the appropriate data subset depending on which task we are running
    task_data = task_subset(data, task_ids, task_idx)
    loader = DataLoader(task_data, batch_size)

    # train
    for _ in tqdm(range(epochs), 'Epochs: '):
        for batch in loader:
            optimizer.zero_grad()
            x, _ = batch
            x = x.to(device)

            loss = model.vae_loss(x, head, len(task_data))
            loss.backward()
            optimizer.step()


def run_generative_task(model, train_data, train_task_ids, test_data, test_task_ids,
                        task_idx, coreset, epochs, batch_size, save_as, device, lr,
                        evaluation_classifier, multiheaded=True, optimizer=None, summary_writer=None):
    """
        Trains a VCL model using online variational inference on a task, and performs a coreset
        training run as well as an evaluation after training.

        :param model: the VCL model to train
        :param train_data: the complete dataset to train on, such as MNIST
        :param train_task_ids: the label-to-task mapping that defines which task examples in the dataset belong to
        :param test_data: the complete dataset to train on, such as MNIST
        :param test_task_ids: the label-to-task mapping that defines which task examples in the dataset belong to
        :param task_idx: task being learned, maps to a specific head in the network
        :param coreset: coreset object to use in training
        :param epochs: number of training epochs
        :param batch_size: batch size used in training
        :param save_as: base directory to save into
        :param device: device to run the experiment on, either 'cpu' or 'cuda'
        :param lr: optimizer learning rate to use
        :param evaluation_classifier: classifier used for the 'classifier uncertainty' test metric
        :param optimizer: optionally, provide an existing optimizer instead of having the method create a new one
        :param multiheaded: true if the network being trained is multi-headed
        :param summary_writer: tensorboard_x summary writer
        """

    print('TASK ', task_idx)

    # separate optimizer for each task
    optimizer = optimizer if optimizer is not None else optim.Adam(model.parameters(), lr=lr)

    head = task_idx if multiheaded else 0

    # obtain correct subset of data for training, and set some aside for the coreset
    task_data = task_subset(train_data, train_task_ids, task_idx)
    non_coreset_data = coreset.select(task_data, task_id=task_idx)
    train_loader = DataLoader(non_coreset_data, batch_size)

    # train
    for epoch in tqdm(range(epochs), 'Epochs: '):
        epoch_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            x, _ = batch
            x = x.to(device)

            loss = model.vae_loss(x, head, len(task_data))
            epoch_loss += len(x) * loss.item()

            loss.backward()
            optimizer.step()

        if summary_writer is not None:
            summary_writer.add_scalars("loss", {"TASK_" + str(task_idx): epoch_loss / len(task_data)}, epoch)

    # after training, prepare for new task by copying posteriors into priors
    model.reset_for_new_task(head)

    # coreset train
    model_cs_trained = coreset.coreset_train(model, optimizer, task_idx, epochs,
                                             device, multiheaded=multiheaded)

    task_confusions = []
    task_likelihoods = []

    for test_task_idx in range(task_idx + 1):
        head = test_task_idx if multiheaded else 0

        # first test using classifier confusion metric
        y_true = torch.zeros(size=(batch_size, 10))
        y_true[:, task_idx] = 1

        x_generated = model_cs_trained.generate(batch_size, head)
        y_pred = evaluation_classifier(x_generated)
        task_confusions.append(F.kl_div(y_pred, y_true))

        print("After task {} confusion on task {} is {}"
              .format(task_idx, test_task_idx, task_confusions[-1]))

        # then test using log likelihood
        task_data = task_subset(test_data, test_task_ids, test_task_idx)
        x = torch.Tensor([x for x, _ in task_data])
        x = x.to(device)
        x_reconstructed = model(x)
        task_likelihoods.append(bernoulli_log_likelihood(x, x_reconstructed).item())

    if summary_writer is not None:
        task_confusions_dict = dict(zip(["TASK_" + str(i) for i in range(task_idx + 1)], task_confusions))
        test_likelihoods_dict = dict(zip(["TASK_" + str(i) for i in range(task_idx + 1)], task_likelihoods))
        summary_writer.add_scalars("test_confusion", task_confusions_dict, task_idx + 1)
        summary_writer.add_scalars("test_likelihoods", test_likelihoods_dict, task_idx + 1)

    write_as_json(save_as + '/accuracy.txt', task_confusions)
    save_model(model, save_as + '_model_task_' + str(task_idx) + '.pth')
