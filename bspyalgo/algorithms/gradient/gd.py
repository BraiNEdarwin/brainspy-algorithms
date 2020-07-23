# TODO: ''' '''
import torch
import os
from tqdm import trange
#from bspyproc.bspyproc import get_processor
from bspyproc.utils.pytorch import TorchUtils
from bspyalgo.utils.io import save, create_directory, create_directory_timestamp
from bspyalgo.algorithms.gradient.core.data import GDData
from bspyalgo.algorithms.gradient.core.optim import get_optimizer
from bspyalgo.algorithms.gradient.core.losses import choose_loss_function


class GD:
    """
    Trains a neural network given data.
    Inputs and targets is assumed to be partitioned in training and validation sets.
    If saving is needed use key in config file : "results_path": "tmp/output/models/nn_test/"
    @authors: ualegre and hruiz
    """

    def __init__(self, configs, is_main=False):
        self.configs = configs
        self.is_main = is_main
        #self.hyperparams = configs["hyperparameters"]

        # self.init_processor()

    def init_dirs(self, base_dir):
        if 'experiment_name' in self.configs:
            main_folder_name = self.configs['experiment_name']
        else:
            main_folder_name = 'gradient_descent_data'
        if self.is_main:
            base_dir = create_directory_timestamp(base_dir, main_folder_name)
        else:
            base_dir = os.path.join(base_dir, main_folder_name)
            create_directory(base_dir)
        self.base_dir = base_dir
        self.default_output_dir = os.path.join(base_dir, 'reproducibility')
        create_directory(self.default_output_dir)
        if self.configs['checkpoints']['use_checkpoints']:
            self.default_checkpoints_dir = os.path.join(base_dir, 'checkpoints')
            create_directory(self.default_checkpoints_dir)

    # def init_loss(self, loss_fn):
    #     if 'loss_function' in self.hyperparams.keys():
    #         self.loss_fn = choose_loss_function(self.hyperparams['loss_function'])
    #     else:
    #         self.loss_fn = loss_fn

    # def calculate_loss(model, model, predictions, targets, use_regularizer=False):
    #     if 'regularizer' in dir(self.processor):
    #         self.loss_fn(predictions, targets) + self.processor.regularizer()
    #     else:
    #         self.criterion = self.loss_fn

    # def reset(self):
    #     self.init_optimizer()
    #     self.processor.reset()

    # def init_optimizer(self, model):
    #    self.optimizer = get_optimizer(filter(lambda p: p.requires_grad, model.parameters()), self.hyperparams)

    # TODO: Move this code to the surrogate model generator
    # def init_processor(self, data_info):
        # TODO: Move this code to the surrogate model generator
        if data_info is not None:
            # This case is only used when using the GD for creating a new model
            print('Using the Gradient Descent for Surrogate Model Generation.')
            try:
                if self.processor.info is not None:
                    print('The model is being retrained as a surrogate model')
                    self.processor.info['data_info_retrain'] = data_info
                    self.processor.info['smg_configs_retrain'] = self.configs
            except AttributeError:
                self.processor.info = {}
                print('The model has been generated from scratch as a torch_model')
                self.processor = TorchUtils.format_tensor(self.processor)
                self.processor.info['data_info'] = data_info
                self.processor.info['smg_configs'] = self.configs
     # TODO: Implement feeding the validation_data and mask as optional kwargs

    def init_dirs(self, save_data):
        if save_data and 'results_base_dir' in self.configs:
            self.init_dirs(self.configs['results_base_dir'])
        self.reset()
        if 'debug' in self.processor.configs and self.processor.configs['debug'] and self.processor.configs['architecture'] == 'device_architecture':
            self.processor.init_dirs(self.configs['results_base_dir'])

    def process_data(self, dataset):
        if self.configs['data']['validation']['use_validation']:
            train_set, val_set = torch.utils.data.random_split(dataset, self.configs['data']['validation']['split'])
            train_set = torch.utils.data.DataLoader(train_set)
            val_set = torch.utils.data.DataLoader(val_set)
        else:
            train_set = torch.utils.data.DataLoader(dataset)
            val_set = None
        return train_set, val_set

    def fit_generator(self, model, dataset, optimizer, loss, data_info=None, save_data=True):
        # data_split = [50000, 10000]

        # Move this function to the SMG
        # self.init_processor(data_info)  # Move this function to the SMG
        self.init_dirs(save_data)

        training_loader, validation_loader = self.process_data(dataset)
        self.train(model, training_loader, validation_loader)

        if save_data:
            save('configs', file_path=os.path.join(self.default_output_dir, 'configs.json'), data=self.configs)
            save('torch', file_path=os.path.joitrain_setn(self.default_output_dir, 'model.pt'), data=self.processor)
            save(mode='pickle', file_path=os.path.join(self.default_output_dir, 'results.pickle'), data=data.results)
        return data

    def train(self, model, dataloaders, epochs, criterion, optimizer):
        train_losses, val_losses = [], []
        looper = trange(epochs, desc=' Initialising')
        for epoch in looper:
            running_loss = 0

            for inputs, targets in dataloaders[0]:
                #inputs = inputs.squeeze()
                #targets = targets.squeeze()

                optimizer.zero_grad()

                predictions = model(inputs)
                loss = criterion(predictions, targets)
                if 'regularizer' in dir(model):
                    loss = loss + model.regularizer()

                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            train_losses.append(running_loss / len(dataloaders[0]))
            description = "Epoch: {}/{}.. ".format(epoch + 1, epochs) + "Training Loss: {:.3f}.. ".format(train_losses[-1])

            if dataloaders[1] is not None:
                val_loss = 0
                accuracy = 0

                with torch.no_grad():
                    model.eval()
                    for inputs, targets in dataloaders[1]:
                        predictions = model(inputs)
                        val_loss += criterion(predictions, targets)

                model.train()

                val_losses.append(val_loss / len(dataloaders[1]))
                description = description + "Test Loss: {:.3f}.. ".format(val_losses[-1]) + "Test Accuracy: {:.3f}".format(accuracy / len(valloader))
            looper.set_description(description)

            # Add a save instruction
            if self.stopping_criteria(train_losses, val_losses):
                break

    def stopping_criteria(self, train_losses, val_losses):
        return False
    # def sgd_train_with_validation(self, data):
    #     x_train = data.results['inputs']
    #     y_train = data.results['targets']
    #     x_val = data.results['inputs_val']
    #     y_val = data.results['targets_val']
    #     looper = trange(self.hyperparams['nr_epochs'], desc=' Initialising')
    #     for epoch in looper:
    #         self.train_step(x_train, y_train)
    #         # with torch.no_grad():
    #         data.results['performance_history'][epoch, 0], prediction_training, data.results['target_indices'] = self.evaluate_training_error(x_val, x_train, y_train)
    #         data.results['performance_history'][epoch, 1], prediction_validation = self.evaluate_validation_error(x_val, y_val)
    #         if self.configs['checkpoints']['use_checkpoints'] and ((epoch + 1) % self.configs['checkpoints']['save_interval'] == 0):
    #             save('torch', os.path.join(self.default_checkpoints_dir, f'checkpoint.pt'), data=self.processor)
    #     #    if epoch % self.hyperparams['save_interval'] == 0:
    #         training_error = data.results['performance_history'][epoch, 0]
    #         validation_error = data.results['performance_history'][epoch, 1]
    #         description = ' Epoch: ' + str(epoch) + ' Training Error:' + str(training_error) + ' Val. Error:' + str(validation_error)
    #         looper.set_description(description)
    #         if training_error <= self.hyperparams['stop_threshold'] and validation_error <= self.hyperparams['stop_threshold']:
    #             print(f"Reached threshold error {self.hyperparams['stop_threshold'] } for training and validation. Stopping")
    #             break
    #     data.set_result_as_numpy('best_output', prediction_validation)
    #     data.set_result_as_numpy('best_output_training', prediction_training)
    #     return data

    # def sgd_train_without_validation(self, data):
    #     x_train = data.results['inputs']
    #     y_train = data.results['targets']
    #     looper = trange(self.hyperparams['nr_epochs'], desc='Initialising')
    #     for epoch in looper:
    #         # self.processor.train()
    #         self.train_step(x_train, y_train)
    #         # self.processor.eval()
    #         with torch.no_grad():
    #             prediction = self.processor(data.results['inputs'])
    #             data.results['performance_history'][epoch] = self.loss_fn(prediction, y_train).item()  # data.results['targets']).item()
    #         if self.configs['checkpoints']['use_checkpoints'] is True and ((epoch + 1) % self.configs['checkpoints']['save_interval'] == 0):
    #             save('torch', os.path.join(self.default_checkpoints_dir, f'checkpoint.pt'), data=self.processor)
    #         # if epoch % self.hyperparams['save_interval'] == 0:
    #         error = data.results['performance_history'][epoch]
    #         description = ' Epoch: ' + str(epoch) + ' Training Error:' + str(error)
    #         looper.set_description(description)
    #         if error <= self.hyperparams['stop_threshold']:
    #             print(f"Reached threshold error {self.hyperparams['stop_threshold']}. Stopping")
    #             break
    #     data.set_result_as_numpy('best_output', prediction)
    #     return data

    # def train_step(self, x_train, y_train):
    #     self.processor.train()
    #     permutation = torch.randperm(x_train.size()[0])  # Permute indices

    #     for mb in range(0, len(permutation), self.hyperparams['batch_size']):
    #         self.minibatch_step(x_train, y_train, permutation[mb:mb + self.hyperparams['batch_size']])

    # def minibatch_step(self, x_train, y_train, indices):
    #     # Get y_pred
    #     x_mb = x_train[indices]
    #     y_pred = self.processor(x_mb)

    #     loss = self.loss_function(y_pred, y_train[indices])

    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     self.optimizer.step()

    # def evaluate_validation_error(self, x_val, y_val):
    #     # Evaluate Validation error
    #     # with torch.no_grad():
    #     self.processor.eval()
    #     prediction = self.processor(x_val)
    #     return self.loss_fn(prediction, y_val).item(), prediction

    # def evaluate_training_error(self, x_val, x_train, y_train):
    #     # Evaluate training error
    #     self.processor.eval()
    #     samples = len(x_val)
    #     target_indices = torch.randperm(len(x_train))[:samples]
    #     x_sampled = x_train[target_indices]
    #     with torch.no_grad():
    #         prediction = self.processor(x_sampled)
    #     target = y_train[target_indices]
    #     return self.loss_fn(prediction, target).item(), prediction, target_indices
