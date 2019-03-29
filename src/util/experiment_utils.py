import torch
from util.operations import task_subset, class_accuracy
from util.outputs import write_as_json, save_model
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


def run_task(model, train_data, train_task_ids,
                test_data, test_task_ids,
                task_idx, optimizer, coreset, epochs, batch_size,
                save_as, y_transform=None, multiheaded=True):

    print('TASK', task_idx)

    head = task_idx if multiheaded else 0

    task_data = task_subset(train_data, train_task_ids, task_idx)
    non_coreset_data = coreset.select(task_data, task_id=task_idx)
    train_loader = DataLoader(non_coreset_data, batch_size)

    for _ in tqdm(range(epochs), 'Epochs: '):
        for batch in train_loader:
            optimizer.zero_grad()
            x, y_true = batch

            if y_transform is not None:
                y_true = y_transform(y_true, task_idx)

            loss = model.loss(x, y_true, head)
            loss.backward()
            optimizer.step()

    # test
    print('Training Coreset')
    model_cs_trained = coreset.coreset_train(model, optimizer, task_idx, epochs, y_transform=y_transform, 
                                             multiheaded=multiheaded)
    task_accuracies = []
    for test_task_idx in range(task_idx+1):
        head = test_task_idx if multiheaded else 0

        task_data = task_subset(test_data, test_task_ids, test_task_idx)

        x      = torch.Tensor([x for x, _ in task_data])
        y_true = torch.Tensor([y for _, y in task_data])

        if y_transform is not None:
            y_true = y_transform(y_true, test_task_idx)

        y_pred = model_cs_trained.prediction(x, head)

        acc = class_accuracy(y_pred, y_true)
        print("After task {} perfomance on task {} is {}"
                .format(task_idx, test_task_idx, acc))
        print()

        task_accuracies.append(acc)

    write_as_json(save_as + '/accuracy.txt', task_accuracies)
    save_model(model, save_as + '_model_task_' + str(task_idx) + '.pth')
