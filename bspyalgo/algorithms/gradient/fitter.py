import torch
from bspyalgo.utils.performance import accuracy
from tqdm import trange


def train(model, dataloaders, epochs, criterion, optimizer, logger=None):
    train_losses, val_losses = [], []
    looper = trange(epochs, desc=' Initialising')
    for epoch in looper:
        running_loss = 0
        val_loss = 0
        for inputs, targets in dataloaders[0]:
            #inputs = inputs.squeeze()
            #targets = targets.squeeze()

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
        description = "Training Loss: {:.3f}.. ".format(train_losses[-1])

        if dataloaders[1] is not None:
            with torch.no_grad():
                model.eval()
                for inputs, targets in dataloaders[1]:
                    predictions = model(inputs)
                    if logger is not None and 'log_ios_val' in dir(logger):
                        logger.log_ios_val(inputs, targets, predictions)
                    val_loss += criterion(predictions, targets)

            model.train()

            val_losses.append(val_loss / len(dataloaders[1]))
            description = description + "Test Loss: {:.3f}.. ".format(val_losses[-1])

        looper.set_description(description)
        if logger is not None and 'log_val_predictions' in dir(logger):
            logger.log_performance(train_losses, val_losses, epoch)

        # TODO: Add a save instruction and a stopping criteria
        # if stopping_criteria(train_losses, val_losses):
        #     break
    if logger is not None:
        logger.close()
    return model, train_losses, val_losses


def test(model, dataset):
    with torch.no_grad():
        model.eval()
        inputs, targets = dataset[:]
        predictions = model(inputs)
    #plot_gate('[ 0 0 0 1]', True, predictions, targets, show_plots=True)
    return accuracy(predictions.squeeze(), targets.squeeze(), plot=None, return_node=True)
