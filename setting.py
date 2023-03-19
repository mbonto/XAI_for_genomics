# Paths
def set_path():
    path = '/local/mbontono/data/'  # absolute path of the data folder
    return path


# Sets of variables
import os
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

def set_optimizer(name, model):
    if name == 'pancan':
        n_epoch = 25
        criterion = nn.CrossEntropyLoss()
        lr = 0.1
        weight_decay = 1e-4
        lr_gamma = 0.1
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        scheduler = MultiStepLR(optimizer, milestones=[int(0.5 * n_epoch), int(0.9 * n_epoch)], gamma=lr_gamma)
    else:
        n_epoch = 25
        criterion = nn.CrossEntropyLoss()
        lr = 0.1
        weight_decay = 1e-4
        lr_gamma = 0.1
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        scheduler = MultiStepLR(optimizer, milestones=[int(0.4 * n_epoch), int(0.7 * n_epoch), int(0.95 * n_epoch)], gamma=lr_gamma)
    return criterion, optimizer, scheduler, n_epoch


def get_save_path(name, code_path):
    if name == 'pancan':
        save_path = os.path.join(code_path, 'Pancan', 'Results')
    else:
        save_path = os.path.join(code_path, 'Simulation', 'Results', name)
    return save_path


def get_data_path(name):
    data_path = set_path()
    if name == 'pancan':
        data_path = os.path.join(data_path, 'tcga')
    else:
        data_path = os.path.join(data_path, 'simulation')
    return data_path