

#     # import pickle as pkl

#     # data_dict = pkl.load(open("tmp/input/best_output_ring_example.pkl", 'rb'))
#     # BEST_OUTPUT = data_dict['best_output']
#     # TARGETS = np.zeros_like(BEST_OUTPUT)
#     # TARGETS[int(len(BEST_OUTPUT) / 2):] = 1
#     # ACCURACY, LABELS, THRESHOLD = perceptron(BEST_OUTPUT, TARGETS, plot='show')

#     # MASK = np.ones_like(TARGETS, dtype=bool)
#     # ACC = accuracy(BEST_OUTPUT, TARGETS)
#     # print(f'Accuracy for best output: {ACC}')
#     import os
#     import matplotlib.pyplot as plt

#     arch = 'single'
#     if arch == 'multiple':
#         main_folder = '/home/unai/Documents/3-programming/brainspy-tasks/tmp/output/ring_nips/searcher_0.00625mV_2020_04_14_181433_multiple/validation.old/validation_2020_04_15_110246_training_data'
#         test_data = np.load(os.path.join(main_folder, 'new_test_50_run_acc_data.npz'))

#         test_outputs = test_data['outputs']
#         test_targets = np.zeros_like(test_outputs)
#         for i in range(test_outputs.shape[1]):
#             test_targets[:, i] = test_data['targets']

#         train_data = np.load(os.path.join(main_folder, 'validation_plot_data.npz'))
#         train_output = train_data['real_output'][train_data['mask']]
#         train_targets_npz = np.load(os.path.join(main_folder, 'targets.npz'))
#         train_targets = 1 - train_targets_npz['masked_targets']

#         acc_train, node_train = accuracy(train_output, train_targets, plot=os.path.join(main_folder, 'train_acc.jpg'), return_node=True)
#         acc_test_single = accuracy(test_outputs[:, 0], test_targets[:, 0], plot=os.path.join(main_folder, 'test_acc_single.jpg'), node=node_train, return_node=False)
#         acc_test_50 = accuracy(test_outputs.flatten(), test_targets.flatten(), plot=os.path.join(main_folder, 'test_acc_50.jpg'), node=node_train, return_node=False)
#     if arch == 'single':
#         import os
#         main_folder = '/home/unai/Documents/3-programming/brainspy-tasks/tmp/output/ring_nips/searcher_0.00625mV_2020_04_16_183324_single_newtrial/validation_hw_trainset/validation_2020_04_16_225147'

#         targets_data = np.load(os.path.join(main_folder, 'targets.npz'))

#         test = np.load(os.path.join(main_folder, 'outputs.npz'))
#         test_outputs = test['hardware_outputs'][test['mask']]
#         test_targets = np.zeros_like(test_outputs)
#         for i in range(test_outputs.shape[1]):
#             test_targets[:, i] = targets_data['test']

#         train = np.load(os.path.join(main_folder, 'validation_plot_data.npz'))
#         train_output = train['real_output'][train['mask']]
#         train_targets = 1 - targets_data['train']

#         acc_train, node_train = accuracy(train_output, train_targets, plot=os.path.join(main_folder, 'train_acc.jpg'), return_node=True)
#         acc_test_single = accuracy(test_outputs[:, 0], test_targets[:, 0], plot=os.path.join(main_folder, 'test_acc_single.jpg'), node=node_train, return_node=False)
#         acc_test_50 = accuracy(test_outputs.flatten(), test_targets.flatten(), plot=os.path.join(main_folder, 'test_acc_50.jpg'), node=node_train, return_node=False)

#         print(0)
