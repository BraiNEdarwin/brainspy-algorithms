#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 11:33:38 2019

@author: hruiz
"""

import numpy as np
import torch.nn as nn
import torch
from bspyalgo.utils.pytorch import TorchModel

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DNPU(TorchModel):
    '''
    '''

    def __init__(self, in_list,
                 path=r'./tmp/NN_model/checkpoint3000_02-07-23h47m.pt'):
        super().__init__()

        self.in_list = in_list
        self.nr_inputs = len(in_list)
        self.load_model(path)

        # Freeze parameters
        for params in self.parameters():
            params.requires_grad = False
        # Define learning parameters
        self.nr_electodes = len(self.info['offset'])
        self.indx_cv = np.delete(np.arange(self.nr_electodes), in_list)
        self.nr_cv = len(self.indx_cv)
        offset = self.info['offset']
        amplitude = self.info['amplitude']

        self.min_voltage = offset - amplitude
        self.max_voltage = offset + amplitude
        bias = self.min_voltage[self.indx_cv] + \
            (self.max_voltage[self.indx_cv] - self.min_voltage[self.indx_cv]) * \
            np.random.rand(1, self.nr_cv)

        bias = torch.as_tensor(bias, dtype=torch.float32).to(DEVICE)
        self.bias = nn.Parameter(bias)
        # Set as torch Tensors and send to DEVICE
        self.indx_cv = torch.tensor(self.indx_cv, dtype=torch.int64).to(DEVICE)  # IndexError: tensors used as indices must be long, byte or bool tensors
        self.amplification = torch.tensor(self.info['amplification']).to(DEVICE)
        self.min_voltage = torch.tensor(self.min_voltage, dtype=torch.float32).to(DEVICE)
        self.max_voltage = torch.tensor(self.max_voltage, dtype=torch.float32).to(DEVICE)

    def forward(self, x):

        expand_cv = self.bias.expand(x.size()[0], -1)
        inp = torch.empty((x.size()[0], x.size()[1] + self.nr_cv)).to(DEVICE)
#        print(x.dtype,self.amplification.dtype)
        inp[:, self.in_list] = x
#        print(inp.dtype,self.indx_cv.dtype,expand_cv.dtype)
        inp[:, self.indx_cv] = expand_cv

        return self.model(inp) * self.amplification

    def regularizer(self):
        low = self.min_voltage[self.indx_cv]
        high = self.max_voltage[self.indx_cv]
#        print(x.dtype,low.dtype,high.dtype)
        assert any(low < 0), \
            "Min. Voltage is assumed to be negative, but value is positive!"
        assert any(high > 0), \
            "Max. Voltage is assumed to be positive, but value is negative!"
        return torch.sum(torch.relu(low - self.bias) + torch.relu(self.bias - high))


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    x = 0.5 * np.random.randn(10, 2)
    x = torch.Tensor(x).to(DEVICE)
    target = torch.Tensor([[5]] * 10).to(DEVICE)

    node = DNPU([0, 4])
    loss = nn.MSELoss()
    optimizer = torch.optim.SGD([{'params': node.parameters()}], lr=0.00005)

    LOSS_LIST = []
    CHANGE_PARAMS_NET = []
    CHANGE_PARAMS0 = []

    START_PARAMS = [p.clone().detach() for p in node.parameters()]

    for eps in range(2000):

        optimizer.zero_grad()
        out = node(x)
        if np.isnan(out.data.cpu().numpy()[0]):
            break
        LOSS = loss(out, target) + node.regularizer()
        LOSS.backward()
        optimizer.step()
        LOSS_LIST.append(LOSS.data.cpu().numpy())
        CURRENT_PARAMS = [p.clone().detach() for p in node.parameters()]
        DELTA_PARAMS = [(current - start).sum() for current, start in zip(CURRENT_PARAMS, START_PARAMS)]
        CHANGE_PARAMS0.append(DELTA_PARAMS[0])
        CHANGE_PARAMS_NET.append(sum(DELTA_PARAMS[1:]))

    END_PARAMS = [p.clone().detach() for p in node.parameters()]
    print("CV params at the beginning: \n ", START_PARAMS[0])
    print("CV params at the end: \n", END_PARAMS[0])
    print("Example params at the beginning: \n", START_PARAMS[-1][:8])
    print("Example params at the end: \n", END_PARAMS[-1][:8])
    print("Length of elements in node.parameters(): \n", [len(p) for p in END_PARAMS])
    print("and their shape: \n", [p.shape for p in END_PARAMS])
    print(f'OUTPUT: \n {out.data.cpu()}')

    plt.figure()
    plt.plot(LOSS_LIST)
    plt.title("Loss per epoch")
    plt.show()
    plt.figure()
    plt.plot(CHANGE_PARAMS0)
    plt.plot(CHANGE_PARAMS_NET)
    plt.title("Difference of parameter with initial params")
    plt.legend(["CV params", "Net params"])
    plt.show()
