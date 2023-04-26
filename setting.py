# Paths
def set_path():
    path = '/local/mbontono/data/'  # absolute path of the data folder
    return path


# Sets of variables
import os
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR


def get_hyperparameters(name, model_name):
    n_layer = None
    n_hidden_feat = None
    if model_name == "MLP":
        if name == "KIRC":
            n_layer = 1
            n_hidden_feat = 20
        elif name == "BRCA":
            n_layer = 1
            n_hidden_feat = 40
        elif name == "s2":
            n_layer = 1
            n_hidden_feat = 20
        elif name == "s1":
            n_layer = 1
            n_hidden_feat = 20
        elif name == "s3":
            n_layer = 1
            n_hidden_feat = 10
        elif name == "s4":
            n_layer = 1
            n_hidden_feat = 10
        elif name == "s5":
            n_layer = 1
            n_hidden_feat = 10
        elif name in ["pancan", "SIMU1", "SIMU2"]:
            n_layer = 1
            n_hidden_feat = 20
        elif name.split("_")[0] == "syn":
            n_layer = 1
            n_hidden_feat = 20
    elif model_name == "DiffuseMLP":
        n_layer = 1
        n_hidden_feat = 20
    return n_layer, n_hidden_feat
        

def set_optimizer(name, model):
    n_epoch = 25
    criterion = nn.CrossEntropyLoss()
    lr = 0.1
    weight_decay = 1e-4
    lr_gamma = 0.1
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=[int(0.5 * n_epoch), int(0.9 * n_epoch)], gamma=lr_gamma)
    return criterion, optimizer, scheduler, n_epoch


def get_save_path(name, code_path):
    if name == 'pancan':
        save_path = os.path.join(code_path, 'Pancan', 'Results')
    elif name == 'BRCA':
        save_path = os.path.join(code_path, 'Gdc', 'Results', 'BRCA')
    elif name == 'KIRC':
        save_path = os.path.join(code_path, 'Gdc', 'Results', 'KIRC')
    elif name == 'LUAD':
        save_path = os.path.join(code_path, 'Gdc', 'Results', 'LUAD')
    else:
        save_path = os.path.join(code_path, 'Simulation', 'Results', name)
    return save_path


def get_data_path(name):
    data_path = set_path()
    if name in ["pancan", "BRCA", "KIRC",]:
        data_path = os.path.join(data_path, 'tcga')
    else:
        data_path = os.path.join(data_path, 'simulation')
    return data_path


def get_setting(name):
    assert name in ["pancan", "BRCA", "KIRC"], "Name should be pancan, BRCA or KIRC"
    if name == "pancan":
        database = "pancan"
        label_name = "type"
        log = True
        _sum = False
    elif name in ["BRCA", "KIRC"]:
        database = "gdc"
        label_name = "sample_type.samples"
        log = True
        _sum = True
    return database, label_name, log, _sum
