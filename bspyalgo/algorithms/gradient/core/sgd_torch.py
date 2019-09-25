#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 17:18:34 2019
Trains a neural network given data.
---------------
Arguments
data :  List containing 2 tuples; the first with a training set (inputs,targets),
        the second with validation data. Both the inputs and targets must be
        torch.Tensors (shape: nr_samplesXinput_dim, nr_samplesXoutput_dim).
network : The network to be trained
conf_dict : Configuration dictionary with hyper parameters for training
save_dir (kwarg, str)  : Path to save the results
---------------
Returns:
network (torch.nn.Module) : trained network
costs (np.array)    : array with the costs (training,validation) per epoch

Notes:
    1) The dopantNet is composed by a surrogate model of a dopant network device
    and bias learnable parameters that serve as control inputs to tune the
    device for desired functionality. If you have this use case, you can get the
    control voltage parameters via network.parameters():
        params = [p.clone().detach() for p in network.parameters()]
        control_voltages = params[0]
    2) For training the surrogate model, the outputs must be scaled by the
    amplification. Hence, the output of the model and the errors are  NOT in nA.
    To get the errors in nA, scale by the amplification**2.
    The dopant network already outputs the prediction in nA. To get the output
    of the surrogate model in nA, use the method .outputs(inputs).

@author: hruiz
"""

import torch
from bspyalgo.utils.io import save, create_directory_timestamp
import numpy as np

# if dir_path and (epoch + 1) % SGD_CONFIGS['save_interval'] == 0:
#             save('torch', config_dict, dir_path, f'checkpoint_epoch{epoch}.pt', torch_model=network)

#         if epoch % 10 == 0:
#             print('Epoch:', epoch,
#                   'Val. Error:', costs[epoch, 1],
#                   'Training Error:', costs[epoch, 0])

#     if dir_path:
#         save('torch', config_dict, dir_path, 'trained_network.pt', torch_model=network)
#     return costs


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    from bspyalgo.algorithms.gradient.core.dopanet import DNPU
    # Get device
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Create data
    in_list = [0, 3]
    x = 0.5 * np.random.randn(10, len(in_list))
    inp_train = torch.Tensor(x).to(DEVICE)
    t_train = torch.Tensor(5. * np.ones((10, 1))).to(DEVICE)
    x = 0.5 * np.random.randn(4, len(in_list))
    inp_val = torch.Tensor(x).to(DEVICE)
    t_val = torch.Tensor(5. * np.ones((4, 1))).to(DEVICE)
    DATA = [(inp_train, t_train), (inp_val, t_val)]
    # Start the node
    node = DNPU(in_list)
    START_PARAMS = [p.clone().detach() for p in node.parameters()]
    # Make config dict
    SGD_CONFIGS = {}
    SGD_CONFIGS['nr_epochs'] = 300
    SGD_CONFIGS['batch_size'] = len(t_train)
    SGD_CONFIGS['learning_rate'] = 3e-5
    SGD_CONFIGS['results_path'] = 'tmp/NN_test/'
    SGD_CONFIGS['experiment_name'] = 'TEST'
    SGD_CONFIGS['save_interval'] = 10
    # NOTE: the values above are for the purpose of the toy problem here and
    #       should not be used elsewere.
    # The default values in the config_dict should be:
    # learning_rate = 1e-4
    # batch_size = 128, save_dir = 'tmp/...',
    # save_interval = 10

    # Train the node
    COSTS = trainer(DATA, node, SGD_CONFIGS)

    OUTPUT = node(inp_val).data.cpu()
    END_PARAMS = [p.clone().detach() for p in node.parameters()]
    print("CV params at the beginning: \n ", START_PARAMS[0])
    print("CV params at the end: \n", END_PARAMS[0])
    print("Example params at the beginning: \n", START_PARAMS[-1][:8])
    print("Example params at the end: \n", END_PARAMS[-1][:8])
    print("Length of elements in node.parameters(): \n", [len(p) for p in END_PARAMS])
    print("and their shape: \n", [p.shape for p in END_PARAMS])
    print(f'OUTPUT: {OUTPUT}')

    plt.figure()
    plt.plot(COSTS)
    plt.title("Loss per epoch")
    plt.legend(["Training", "Validation"])
    plt.show()
