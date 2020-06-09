#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 11:42:27 2018

@author: hruiz
"""
from __future__ import generator_stop
import numpy as np
import torch.nn as nn
import torch
from matplotlib import pyplot as plt
from more_itertools import grouper
from tqdm import trange
from bspyproc.utils.pytorch import TorchUtils


def batch_generator(nr_samples, batch):
    batches = grouper(np.random.permutation(nr_samples), batch)
    while True:
        try:
            indices = list(next(batches))
            if None in indices:
                indices = [index for index in indices if index is not None]
            yield torch.tensor(indices, dtype=torch.int64)
        except StopIteration:
            return


def decision(data, targets, lrn_rate=0.0007, mini_batch=8, max_iters=100, validation=False, verbose=True):

    if validation:
        n_total = len(data)
        assert n_total > 10, "Not enough data, we assume you have at least 10 points"
        n_val = int(n_total * 0.1)
        shuffle = np.random.permutation(n_total)
        indices_train = shuffle[n_val:]
        indices_val = shuffle[:n_val]
        x_train = torch.tensor(data[indices_train], dtype=TorchUtils.data_type)
        t_train = torch.tensor(targets[indices_train], dtype=TorchUtils.data_type)
        x_val = torch.tensor(data[indices_val], dtype=TorchUtils.data_type)
        t_val = torch.tensor(targets[indices_val], dtype=TorchUtils.data_type)
    else:
        x_train = torch.tensor(data, dtype=TorchUtils.data_type)
        t_train = torch.tensor(targets, dtype=TorchUtils.data_type)
        x_val = torch.tensor(data, dtype=TorchUtils.data_type)
        t_val = torch.tensor(targets, dtype=TorchUtils.data_type)

    node = nn.Linear(1, 1).type(TorchUtils.data_type)
    loss = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(node.parameters(), lr=lrn_rate, betas=(0.999, 0.999))
    best_accuracy = -1
    looper = trange(max_iters, desc='Calculating accuracy')
    for epoch in looper:
        for mb in batch_generator(len(x_train), mini_batch):
            x_i, t_i = x_train[mb], t_train[mb]
            optimizer.zero_grad()
            y_i = node(x_i)
            cost = loss(y_i, t_i)
            cost.backward()
            optimizer.step()
        with torch.no_grad():
            y = node(x_val)
            labels = y > 0.
            correct_labeled = torch.sum(labels == t_val).detach().numpy()
            acc = 100. * correct_labeled / len(t_val)
            if acc > best_accuracy:
                best_accuracy = acc
                with torch.no_grad():
                    w, b = [p.detach().numpy() for p in node.parameters()]
                    decision_boundary = -b / w
                    predicted_class = node(torch.tensor(data, dtype=TorchUtils.data_type)).detach().numpy() > 0.
        if verbose:
            looper.set_description(f' Epoch: {epoch}  Accuracy {acc}, loss: {cost.item()}')

    return best_accuracy, predicted_class, decision_boundary, node


def decision_pretrained(data, targets, node, validation=False, verbose=True):
    if validation:
        n_total = len(data)
        assert n_total > 10, "Not enough data, we assume you have at least 10 points"
        n_val = int(n_total * 0.1)
        shuffle = np.random.permutation(n_total)
        indices_train = shuffle[n_val:]
        indices_val = shuffle[:n_val]
        x_val = torch.tensor(data[indices_val], dtype=TorchUtils.data_type)
        t_val = torch.tensor(targets[indices_val], dtype=TorchUtils.data_type)
    else:
        x_val = torch.tensor(data, dtype=TorchUtils.data_type)
        t_val = torch.tensor(targets, dtype=TorchUtils.data_type)
    with torch.no_grad():
        w, b = [p.detach().numpy() for p in node.parameters()]
        decision_boundary = -b / w

    best_accuracy = -1
    # looper = trange(max_iters, desc='Calculating accuracy')
    with torch.no_grad():
        y = node(x_val)
        labels = y > 0.
        correct_labeled = torch.sum(labels == t_val).detach().numpy()
        acc = 100. * correct_labeled / len(t_val)
        if acc > best_accuracy:
            best_accuracy = acc
            with torch.no_grad():
                w, b = [p.detach().numpy() for p in node.parameters()]
                decision_boundary = -b / w
                predicted_class = node(torch.tensor(data, dtype=TorchUtils.data_type)).detach().numpy() > 0.
        # if verbose:
        #    looper.set_description(f' Epoch: {epoch}  Accuracy {acc}, loss: {cost.item()}')

    return best_accuracy, predicted_class, decision_boundary, node


def perceptron(input_waveform, target_waveform, plot=None, node=None):
    # Assumes that the input_waveform and the target_waveform have the shape (n_total,1)
    # Normalizes the data; it is assumed that the target_waveform has binary values
    if isinstance(input_waveform, torch.Tensor):
        input_waveform = input_waveform.detach().cpu().numpy()

    original_input_waveform = input_waveform.copy()
    input_waveform = (input_waveform - np.mean(input_waveform, axis=0)) / np.std(input_waveform, axis=0)
    if node is None:
        _accuracy, predicted_labels, threshold, node = decision(input_waveform, target_waveform)
    else:
        _accuracy, predicted_labels, threshold, _ = decision_pretrained(input_waveform, target_waveform, node)
    if plot:
        plt.figure()
        plt.title(f'Accuracy: {_accuracy:.2f} %')
        plt.plot(input_waveform, label='Norm. Waveform')
        plt.plot(predicted_labels, '.', label='Predicted labels')
        plt.plot(target_waveform, 'g', label='Targets')
        plt.plot(np.arange(len(predicted_labels)),
                 np.ones_like(predicted_labels) * threshold, 'k:', label='Threshold')
        plt.legend()
        if plot == 'show':
            plt.show()
        else:
            np.savez(plot + 'accuracy_results', original_input_waveform=original_input_waveform, norm_input_waveform=input_waveform, predicted_labels=predicted_labels, target_waveform=target_waveform, threshold=threshold, accuracy=_accuracy)
            plt.savefig(plot)
            plt.close()
    return _accuracy, predicted_labels, threshold, node


def corr_coeff(x, y):
    return np.corrcoef(np.concatenate((x, y), axis=0))[0, 1]

# TODO: use data object to get the accuracy (see corr_coeff above)


def accuracy(best_output, target_waveforms, plot=None, node=None, return_node=False):
    if len(best_output.shape) == 1:
        y = best_output[:, np.newaxis]
    else:
        y = best_output
    if len(target_waveforms.shape) == 1:
        trgt = target_waveforms[:, np.newaxis]
    else:
        trgt = target_waveforms
    if node is None:
        acc, _, _, node = perceptron(y, trgt, plot=plot)
    else:
        acc, _, _, _ = perceptron(y, trgt, plot=plot, node=node)
    if return_node:
        return acc, node
    else:
        return acc


if __name__ == '__main__':

    # import pickle as pkl

    # data_dict = pkl.load(open("tmp/input/best_output_ring_example.pkl", 'rb'))
    # BEST_OUTPUT = data_dict['best_output']
    # TARGETS = np.zeros_like(BEST_OUTPUT)
    # TARGETS[int(len(BEST_OUTPUT) / 2):] = 1
    # ACCURACY, LABELS, THRESHOLD = perceptron(BEST_OUTPUT, TARGETS, plot='show')

    # MASK = np.ones_like(TARGETS, dtype=bool)
    # ACC = accuracy(BEST_OUTPUT, TARGETS)
    # print(f'Accuracy for best output: {ACC}')
    import os
    import matplotlib.pyplot as plt

    arch = 'single'
    if arch == 'multiple':
        main_folder = '/home/unai/Documents/3-programming/brainspy-tasks/tmp/output/ring_nips/searcher_0.00625mV_2020_04_14_181433_multiple/validation.old/validation_2020_04_15_110246_training_data'
        test_data = np.load(os.path.join(main_folder, 'new_test_50_run_acc_data.npz'))

        test_outputs = test_data['outputs']
        test_targets = np.zeros_like(test_outputs)
        for i in range(test_outputs.shape[1]):
            test_targets[:, i] = test_data['targets']

        train_data = np.load(os.path.join(main_folder, 'validation_plot_data.npz'))
        train_output = train_data['real_output'][train_data['mask']]
        train_targets_npz = np.load(os.path.join(main_folder, 'targets.npz'))
        train_targets = 1 - train_targets_npz['masked_targets']

        acc_train, node_train = accuracy(train_output, train_targets, plot=os.path.join(main_folder, 'train_acc.jpg'), return_node=True)
        acc_test_single = accuracy(test_outputs[:, 0], test_targets[:, 0], plot=os.path.join(main_folder, 'test_acc_single.jpg'), node=node_train, return_node=False)
        acc_test_50 = accuracy(test_outputs.flatten(), test_targets.flatten(), plot=os.path.join(main_folder, 'test_acc_50.jpg'), node=node_train, return_node=False)
    if arch == 'single':
        import os
        main_folder = '/home/unai/Documents/3-programming/brainspy-tasks/tmp/output/ring_nips/searcher_0.00625mV_2020_04_16_183324_single_newtrial/validation_hw_trainset/validation_2020_04_16_225147'

        targets_data = np.load(os.path.join(main_folder, 'targets.npz'))

        test = np.load(os.path.join(main_folder, 'outputs.npz'))
        test_outputs = test['hardware_outputs'][test['mask']]
        test_targets = np.zeros_like(test_outputs)
        for i in range(test_outputs.shape[1]):
            test_targets[:, i] = targets_data['test']

        train = np.load(os.path.join(main_folder, 'validation_plot_data.npz'))
        train_output = train['real_output'][train['mask']]
        train_targets = 1 - targets_data['train']

        acc_train, node_train = accuracy(train_output, train_targets, plot=os.path.join(main_folder, 'train_acc.jpg'), return_node=True)
        acc_test_single = accuracy(test_outputs[:, 0], test_targets[:, 0], plot=os.path.join(main_folder, 'test_acc_single.jpg'), node=node_train, return_node=False)
        acc_test_50 = accuracy(test_outputs.flatten(), test_targets.flatten(), plot=os.path.join(main_folder, 'test_acc_50.jpg'), node=node_train, return_node=False)

        print(0)
