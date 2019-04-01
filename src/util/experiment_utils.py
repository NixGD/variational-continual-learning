import torch
from util.operations import task_subset, class_accuracy
from util.outputs import write_as_json, save_model
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

def run_point_estimate_initialisation(model, data, optimizer, epochs, task_ids,
                                      batch_size, device, task_idx=0,
                                      y_transform=None, multiheaded=True):

    print("Obtaining point estimate for posterior initialisation")

    task_data = task_subset(data, task_ids, task_idx)
    loader = DataLoader(task_data, batch_size)

    for _ in tqdm(range(epochs), 'Epochs: '):
        for batch in loader:
            optimizer.zero_grad()
            x, y_true = batch
            x = x.to(device)
            y_true = y_true.to(device)

            if y_transform is not None:
                y_true = y_transform(y_true)

            loss = model.point_estimate_loss(x, y_true, task=task_idx)
            loss.backward()
            optimizer.step()

def run_task(model, train_data, train_task_ids, test_data, test_task_ids,
             task_idx, optimizer, coreset, epochs, batch_size, save_as,
             device, y_transform=None, multiheaded=True, summary_writer=None):

    print('TASK', task_idx)

    head = task_idx if multiheaded else 0

    task_data = task_subset(train_data, train_task_ids, task_idx)
    non_coreset_data = coreset.select(task_data, task_id=task_idx)
    train_loader = DataLoader(non_coreset_data, batch_size)

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

        if (summary_writer != None):
            summary_writer.add_scalars("loss", {"TASK_" + str(task_idx): epoch_loss / len(task_data)}, epoch)

    # test
    model_cs_trained = coreset.coreset_train(model, optimizer, task_idx, epochs,
                                             device, y_transform=y_transform,
                                             multiheaded=multiheaded)
    task_accuracies = []
    for test_task_idx in range(task_idx+1):
        head = test_task_idx if multiheaded else 0

        task_data = task_subset(test_data, test_task_ids, test_task_idx)

        x      = torch.Tensor([x for x, _ in task_data])
        y_true = torch.Tensor([y for _, y in task_data])
        x = x.to(device)
        y_true = y_true.to(device)

        if y_transform is not None:
            y_true = y_transform(y_true, test_task_idx)

        y_pred = model_cs_trained.prediction(x, head)

        acc = class_accuracy(y_pred, y_true)
        print("After task {} perfomance on task {} is {}"
                .format(task_idx, test_task_idx, acc))

        task_accuracies.append(acc)

    if summary_writer != None:
        task_accuracies_dict = dict(zip(["TASK_" + str(i) for i in range(task_idx + 1)], task_accuracies))
        summary_writer.add_scalars("test_accuracy", task_accuracies_dict, task_idx + 1)
        summary_writer.add_scalar("mean_posterior_variance", model._mean_posterior_variance(), task_idx + 1)

    write_as_json(save_as + '/accuracy.txt', task_accuracies)
    save_model(model, save_as + '_model_task_' + str(task_idx) + '.pth')
