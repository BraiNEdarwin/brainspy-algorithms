# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 13:14:52 2019

@author: HCRuiz and Unai Alegre
"""
import torch
import numpy as np
from bspyalgo.algorithms.performance import perceptron
from bspyproc.utils.pytorch import TorchUtils

# TODO: implement corr_lin_fit (AF's last fitness function)?


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


def corrsig_fit(outputpool, target, clipvalue=[-np.inf, np.inf]):
    genomes = len(outputpool)
    fitpool = TorchUtils.format_tensor(torch.zeros(genomes))
    for j in range(genomes):
        output = outputpool[j]
        if torch.any(output < clipvalue[0]) or torch.any(output > clipvalue[1]):
            #print(f'Clipped at {torch.abs(output)} nA')
            fit = -1
        else:
            x = torch.stack((output, target[j]), axis=0).squeeze(dim=2)
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

#### GD ###


def corrsig(output, target):
    corr = torch.mean((output - torch.mean(output)) * (target - torch.mean(target))) / \
        (torch.std(output) * torch.std(target) + 1e-10)
    x_high_min = torch.min(output[(target == 1)])
    x_low_max = torch.max(output[(target == 0)])
    delta = x_high_min - x_low_max
    return (1.1 - corr) / torch.sigmoid((delta - 5) / 3)


def sqrt_corrsig(output, target):
    corr = torch.mean((output - torch.mean(output)) * (target - torch.mean(target))) / \
        (torch.std(output) * torch.std(target) + 1e-10)
    x_high_min = torch.min(output[(target == 1)])
    x_low_max = torch.max(output[(target == 0)])
    delta = x_high_min - x_low_max
    # 5/3 works for 0.2 V gap
    return (1. - corr)**(1 / 2) / torch.sigmoid((delta - 2) / 5)


def fisher(output, target):
    '''Separates classes irrespective of assignments.
    Reliable, but insensitive to actual classes'''
    x_high = output[(target == 1)]
    x_low = output[(target == 0)]
    m0, m1 = torch.mean(x_low), torch.mean(x_high)
    s0, s1 = torch.var(x_low), torch.var(x_high)
    mean_separation = (m1 - m0)**2
    return -mean_separation / (s0 + s1)


def fisher_added_corr(output, target):
    x_high = output[(target == 1)]
    x_low = output[(target == 0)]
    m0, m1 = torch.mean(x_low), torch.mean(x_high)
    s0, s1 = torch.var(x_low), torch.var(x_high)
    mean_separation = (m1 - m0)**2
    corr = torch.mean((output - torch.mean(output)) * (target - torch.mean(target))) / \
        (torch.std(output) * torch.std(target) + 1e-10)
    return (1 - corr) - 0.5 * mean_separation / (s0 + s1)


def fisher_multipled_corr(output, target):
    x_high = output[(target == 1)]
    x_low = output[(target == 0)]
    m0, m1 = torch.mean(x_low), torch.mean(x_high)
    s0, s1 = torch.var(x_low), torch.var(x_high)
    mean_separation = (m1 - m0)**2
    corr = torch.mean((output - torch.mean(output)) * (target - torch.mean(target))) / \
        (torch.std(output) * torch.std(target) + 1e-10)
    return (1 - corr) * (s0 + s1) / mean_separation
