import torch
from bspyalgo.utils.performance import accuracy
from tqdm import trange


def train(model, dataloaders, epochs, criterion, optimizer, logger=None):
    train_losses, val_losses = [], []
    looper = trange(epochs, desc=' Initialising')
    for _ in looper:
        running_loss = 0
        val_loss = 0
        for inputs, targets in dataloaders[0]:
            #inputs = inputs.squeeze()
            #targets = targets.squeeze()

            optimizer.zero_grad()
            if logger is not None and 'log_train_inputs' in dir(logger):
                logger.log_train_inputs(inputs, targets)
            predictions = model(inputs)
            if logger is not None and 'log_train_predictions' in dir(logger):
                logger.log_train_predictions(predictions)
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
                    if logger is not None and 'log_val_inputs' in dir(logger):
                        logger.log_val_inputs(inputs, targets)
                    predictions = model(inputs)
                    if logger is not None and 'log_val_predictions' in dir(logger):
                        logger.log_val_predictions(inputs, targets)
                    val_loss += criterion(predictions, targets)

            model.train()

            val_losses.append(val_loss / len(dataloaders[1]))
            description = description + "Test Loss: {:.3f}.. ".format(val_losses[-1])

        looper.set_description(description)
        if logger is not None and 'log_val_predictions' in dir(logger):
            logger.log_performance(train_losses, val_losses)

        # TODO: Add a save instruction and a stopping criteria
        # if stopping_criteria(train_losses, val_losses):
        #     break
    return model, train_losses, val_losses


def test(model, dataset):
    with torch.no_grad():
        model.eval()
        inputs, targets = dataset[:]
        predictions = model(inputs)

    return accuracy(predictions.squeeze(), targets.squeeze(), plot=None, return_node=True)
