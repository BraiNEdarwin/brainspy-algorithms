# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 13:14:52 2019

@author: HCRuiz and Unai Alegre
"""
import torch
import numpy as np
from bspyalgo.utils.performance import perceptron
from bspyproc.utils.pytorch import TorchUtils
# TODO: implement corr_lin_fit (AF's last fitness function)?


def choose_fitness_function(fitness):
    '''Gets the fitness function used in GA from the module FitnessFunctions
    The fitness functions must take two arguments, the outputs of the black-box and the target
    and must return a numpy array of scores of size len(outputs).
    '''
    if fitness == 'corr_fit':
        return corr_fit
    elif fitness == 'accuracy_fit':
        return accuracy_fit
    elif fitness == 'corrsig_fit':
        return corrsig_fit
    else:
        raise NotImplementedError(f"Fitness function {fitness} is not recognized!")

# %% Accuracy of a perceptron as fitness: meanures separability


def accuracy_fit(outputpool, target, clipvalue=np.inf):
    genomes = len(outputpool)
    fitpool = TorchUtils.format_tensor(torch.zeros(genomes))
    for j in range(genomes):
        output = outputpool[j]

        if torch.any(output < clipvalue[0]) or torch.any(output > clipvalue[1]):
            acc = 0
            # print(f'Clipped at {clipvalue} nA')
        else:
            x = output[:, np.newaxis]
            y = target[:, np.newaxis]
            acc, _, _ = perceptron(x, y)

        fitpool[j] = acc
    return fitpool

# %% Correlation between output and target: measures similarity


def corr_fit(outputpool, target, clipvalue=np.inf):
    genomes = len(outputpool)
    fitpool = TorchUtils.format_tensor(torch.zeros(genomes))
    for j in range(genomes):
        output = outputpool[j]
        if torch.any(output < clipvalue[0]) or torch.any(output > clipvalue[1]):
            # print(f'Clipped at {clipvalue} nA')
            corr = -1
        else:
            x = output[:, np.newaxis]
            y = target[:, np.newaxis]
            X = torch.stack((x, y), axis=0)[:, :, 0]
            corr = corrcoef(X)[0, 1]

        fitpool[j] = corr
    return fitpool

# %% Combination of a sigmoid with pre-defined separation threshold (2.5 nA) and
# the correlation function. The sigmoid can be adapted by changing the function 'sig( , x)'


def corrsig_fit(outputpool, target, clipvalue=np.inf):
    genomes = len(outputpool)
    fitpool = TorchUtils.format_tensor(torch.zeros(genomes))
    for j in range(genomes):
        output = outputpool[j]
        if torch.any(output < clipvalue[0]) or torch.any(output > clipvalue[1]):
            # print(f'Clipped at {torch.abs(output)} nA')
            fit = -1
        else:
            x = torch.stack((output, target), axis=0).squeeze(dim=2)
            corr = corrcoef(x)[0, 1]
            buff0 = target == 0
            buff1 = target == 1
            sep = output[buff1].mean() - output[buff0].mean()
            sig = 1 / (1 + torch.exp(-2 * (sep - 2)))
            fit = corr * sig
        fitpool[j] = fit
    return fitpool


def corrcoef(x):
    """
    Mimics `np.corrcoef`

    Arguments
    ---------
    x : 2D torch.Tensor

    Returns
    -------
    c : torch.Tensor
        if x.size() = (5, 100), then return val will be of size (5,5)
    """
    # calculate covariance matrix of rows
    mean_x = torch.mean(x, 1).unsqueeze(dim=1)
    xm = x.sub(mean_x.expand_as(x))
    c = xm.mm(xm.t())
    c = c / (x.size(1) - 1)

    # normalize covariance matrix
    d = torch.diag(c)
    stddev = torch.pow(d, 0.5)
    c = c.div(stddev.expand_as(c))
    c = c.div(stddev.expand_as(c).t())

    # clamp between -1 and 1
    # probably not necessary but numpy does it
    c = torch.clamp(c, -1.0, 1.0)

    return c
