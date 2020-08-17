#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 11:42:27 2018

@author: hruiz
"""
from __future__ import generator_stop
import numpy as np
from torch import nn
import torch
from matplotlib import pyplot as plt
from more_itertools import grouper
from tqdm import trange
from bspyproc.utils.pytorch import TorchUtils
import os


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


def decision(data, targets, node=None, lrn_rate=0.0007, mini_batch=8, max_iters=100, validation=False, verbose=True):

    if validation:
        n_total = len(data)
        assert n_total > 10, "Not enough data, we assume you have at least 10 points"
        n_val = int(n_total * 0.1)
        shuffle = np.random.permutation(n_total)
        indices_train = shuffle[n_val:]
        indices_val = shuffle[:n_val]
        x_train = data[indices_train]
        t_train = targets[indices_train]
        x_val = data[indices_val]
        t_val = targets[indices_val]
    else:
        x_train = x_val = data
        t_train = t_val = targets
    if node is None:
        train = True
        node = nn.Linear(1, 1)
    else:
        train = False
    node = TorchUtils.format_tensor(node)
    loss = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(node.parameters(), lr=lrn_rate, betas=(0.999, 0.999))
    best_accuracy = -1

    if train:
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
                labels = node(t_val) > 0.
                correct_labelled = torch.sum(labels == targets)
                acc = 100. * correct_labelled / len(targets)
                if acc > best_accuracy:
                    best_accuracy = acc
                    predicted_class, decision_boundary = evaluate_node(node, x_val, t_val, best_accuracy)
            if verbose:
                looper.set_description(f'Epoch: {epoch+1}  Accuracy {best_accuracy}, loss: {cost.item()}')
    else:
        labels = node(t_val) > 0.
        correct_labeled = torch.sum(labels == targets)
        best_accuracy = 100. * correct_labeled / len(targets)
        predicted_class, decision_boundary = evaluate_node(node, x_val, t_val, best_accuracy)
        print('Accuracy: ' + str(best_accuracy.item()))
    return best_accuracy, predicted_class, decision_boundary, node


def evaluate_node(node, inputs, targets, best_accuracy):
    with torch.no_grad():
        w, b = [p for p in node.parameters()]
        decision_boundary = -b / w
        prediction = node(inputs)
        predicted_class = prediction > 0.

    return predicted_class, decision_boundary
# def decision_pretrained(data, targets, node, validation=False, verbose=True):
#     if validation:
#         n_total = len(data)
#         assert n_total > 10, "Not enough data, we assume you have at least 10 points"
#         n_val = int(n_total * 0.1)
#         shuffle = np.random.permutation(n_total)
#         indices_train = shuffle[n_val:]
#         indices_val = shuffle[:n_val]
#         x_val = torch.tensor(data[indices_val], dtype=TorchUtils.data_type)
#         t_val = torch.tensor(targets[indices_val], dtype=TorchUtils.data_type)
#     else:
#         data = x_train = x_val = TorchUtils.format_tensor(torch.tensor(data))
#         targets = t_train = t_val = TorchUtils.format_tensor(targets)
#     with torch.no_grad():
#         w, b = [p.detach().numpy() for p in node.parameters()]
#         decision_boundary = -b / w

#     best_accuracy = -1
#     # looper = trange(max_iters, desc='Calculating accuracy')
#     with torch.no_grad():
#         y = node(x_val)
#         labels = y > 0.
#         correct_labeled = torch.sum(labels == t_val).detach().numpy()
#         acc = 100. * correct_labeled / len(t_val)
#         if acc > best_accuracy:
#             best_accuracy = acc
#             with torch.no_grad():
#                 w, b = [p.detach().numpy() for p in node.parameters()]
#                 decision_boundary = -b / w
#                 predicted_class = node(torch.tensor(data, dtype=TorchUtils.data_type)).detach().numpy() > 0.
#         # if verbose:
#         #    looper.set_description(f' Epoch: {epoch}  Accuracy {acc}, loss: {cost.item()}')

#     return best_accuracy, predicted_class, decision_boundary, node


def perceptron(inputs, targets, node=None):
    # Assumes that the input_waveform and the target_waveform have the shape (n_total,1)
    # Normalizes the data; it is assumed that the target_waveform has binary values
    # if isinstance(input_waveform, torch.Tensor):
    #     input_waveform = input_waveform.detach().cpu().numpy()
    inputs = TorchUtils.format_tensor(inputs)
    targets = TorchUtils.format_tensor(targets)

    results = {}
    assert len(inputs.shape) != 1 and len(targets.shape) != 1, "Please unsqueeze inputs and targets"
    original_input_waveform = inputs.clone()
    inputs = (inputs - torch.mean(inputs, axis=0)) / torch.std(inputs, axis=0)
    _accuracy, predictions, threshold, node = decision(inputs, targets, node)

    results['inputs'] = original_input_waveform
    results['norm_inputs'] = inputs
    results['targets'] = targets
    results['predictions'] = predictions
    results['threshold'] = threshold
    results['node'] = node
    results['accuracy_value'] = _accuracy

    return results  # _accuracy, predictions, threshold, node


def plot_perceptron(results, save_dir=None, show_plot=False):
    fig = plt.figure()
    plt.title(f"Accuracy: {results['accuracy_value']:.2f} %")
    plt.plot(TorchUtils.get_numpy_from_tensor(results['norm_inputs']), label='Norm. Waveform')
    plt.plot(TorchUtils.get_numpy_from_tensor(results['predictions']), '.', label='Predicted labels')
    plt.plot(TorchUtils.get_numpy_from_tensor(results['targets']), 'g', label='Targets')
    plt.plot(np.arange(len(results['predictions'])),
             TorchUtils.get_numpy_from_tensor(torch.ones_like(results['predictions']) * results['threshold']), 'k:', label='Threshold')
    plt.legend()
    if show_plot:
        plt.show()
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, 'accuracy.jpg'))
    plt.close()
    return fig


def corr_coeff(x, y):
    return np.corrcoef(np.concatenate((x, y), axis=0))[0, 1]


def corr_coeff_torch(x, y):
    x = TorchUtils.get_numpy_from_tensor(x)
    y = TorchUtils.get_numpy_from_tensor(y)
    result = np.corrcoef(np.concatenate((x, y), axis=0))[0, 1]
    return TorchUtils.get_tensor_from_numpy(result)

# TODO: use data object to get the accuracy (see corr_coeff above)


# def accuracy(predictions, targets, node=None):
#     # TODO: If it is numpy transform it to torch
#     return perceptron(predictions, targets, node)
