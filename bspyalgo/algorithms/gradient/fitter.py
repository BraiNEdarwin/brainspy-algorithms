import torch
from bspyalgo.utils.performance import accuracy
from tqdm import trange
import numpy as np
import os

from torch.utils.data import SubsetRandomSampler


def train(model, dataloaders, epochs, criterion, optimizer, logger=None, save_dir=None, return_best_model=True):
    train_losses, val_losses = [], []
    min_val_loss = np.inf
    looper = trange(epochs, desc=' Initialising')
    for epoch in looper:
        running_loss = 0
        val_loss = 0
        for inputs, targets in dataloaders[0]:

            optimizer.zero_grad()
            predictions = model(inputs)
            if logger is not None and 'log_ios_train' in dir(logger):
                logger.log_ios_train(inputs, targets, predictions, epoch)
            loss = criterion(predictions, targets)
            if 'regularizer' in dir(model):
                loss = loss + model.regularizer()

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_losses.append(running_loss / len(dataloaders[0]))
        description = "Training Loss: {:.6f}.. ".format(train_losses[-1])

        if dataloaders[1] is not None and len(dataloaders[1]) > 0:
            with torch.no_grad():
                model.eval()
                for inputs, targets in dataloaders[1]:
                    predictions = model(inputs)
                    if logger is not None and 'log_ios_val' in dir(logger):
                        logger.log_ios_val(inputs, targets, predictions)
                    val_loss += criterion(predictions, targets)

            model.train()

            val_losses.append(val_loss / len(dataloaders[1]))
            description += "Test Loss: {:.6f}.. ".format(val_losses[-1])
            if save_dir is not None and val_losses[-1] < min_val_loss:
                min_val_loss = val_losses[-1]
                description += ' Saving model ...'
                torch.save(model, os.path.join(save_dir, 'best_model.pt'))
        looper.set_description(description)
        if logger is not None and 'log_val_predictions' in dir(logger):
            logger.log_performance(train_losses, val_losses, epoch)

        # TODO: Add a save instruction and a stopping criteria
        # if stopping_criteria(train_losses, val_losses):
        #     break

    torch.save(model, os.path.join(save_dir, 'model.pt'))
    if logger is not None:
        logger.close()
    if save_dir is not None and return_best_model and dataloaders[1] is not None and len(dataloaders[1]) > 0:
        model = torch.load(os.path.join(save_dir, 'best_model.pt'))
    return model, [torch.tensor(train_losses), torch.tensor(val_losses)]


# def test(model, dataset):
#     with torch.no_grad():
#         model.eval()
#         inputs, targets = dataset[:]
#         predictions = model(inputs)
#     #plot_gate('[ 0 0 0 1]', True, predictions, targets, show_plots=True)
#     return accuracy(predictions.squeeze(), targets.squeeze(), plot=None, return_node=True)

def split(dataset, batch_size, num_workers, sampler=SubsetRandomSampler, split_percentages=[0.8, 0.1, 0.1]):
    # Split percentages are expected to be in the following format: [80,10,10]
    percentages = np.array(split_percentages)
    assert np.sum(percentages) == 1, 'Split percentage does not sum up to 1'
    indices = list(range(len(dataset)))
    np.random.shuffle(indices)
    max_train_index = int(np.floor(percentages[0] * len(dataset)))
    max_dev_index = int(np.floor((percentages[0] + percentages[1]) * len(dataset)))
    max_test_index = int(np.floor(np.sum(percentages) * len(dataset)))

    train_index = indices[:max_train_index]
    dev_index = indices[max_train_index:max_dev_index]
    test_index = indices[max_dev_index:max_test_index]

    train_sampler = sampler(train_index)
    dev_sampler = sampler(dev_index)
    test_sampler = sampler(test_index)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
    dev_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=dev_sampler, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=test_sampler, num_workers=num_workers)

    return [train_loader, dev_loader, test_loader]  # , [train_index, dev_index, test_loader]
