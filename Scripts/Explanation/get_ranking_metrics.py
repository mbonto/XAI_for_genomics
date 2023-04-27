# Libraries
import os
import sys
code_path = os.path.split(os.path.split(os.getcwd())[0])[0]
sys.path.append(code_path)

import numpy as np
import matplotlib.pyplot as plt
import torch
import argparse
from scipy.sparse import load_npz, csc_matrix

from setting import *
from loader import *
from models import *
from XAI_method import *
from graphs import *
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Arguments
argParser = argparse.ArgumentParser()
argParser.add_argument("-n", "--name", type=str, help="dataset name")
argParser.add_argument("-m", "--model", type=str, help="model name (LR, MLP, DiffuseLR, DiffuseMLP)")
argParser.add_argument("--exp", type=int, help="experiment number", default=1)
argParser.add_argument("--diffusion", help="smooth the attributions", action='store_true')
args = argParser.parse_args()
name = args.name
model_name = args.model
exp = args.exp
diffusion = args.diffusion
print('Model    ', model_name)


# Path
save_path = get_save_path(name, code_path)
data_path = get_data_path(name)
save_name = os.path.join(model_name, f"exp_{exp}")


# Dataset
data = np.load(os.path.join(data_path, f'{name}.npy'), allow_pickle=True).item()
studied_class = np.arange(data['n_class'])
base_class = None
if name in ['SimuB']:
    studied_class = np.array([0, 1, 2])
    base_class = 3
elif name in ['SimuA',]:
    studied_class = np.array([0,])
    base_class = 1
elif name in ['SimuC',]:
    studied_class = np.array([0, 1, 2, 3, 4])
    base_class = 5

# Special case where several subclasses are inside a class: the subgroup labels need to be retrieved
if name in ['SimuA', 'SimuB', 'SimuC'] or name[:5] == "syn_g":
    X = data['X']
    y = data['y']
    ## Create train/test sets
    train_indices, test_indices = split_indices(len(X), test_size=0.4, random_state=43)
    X_train, X_test, y_train, y_test = split_data_from_indices(X, y, train_indices, test_indices)
    X_train = torch.from_numpy(X_train).type(torch.float)
    X_test = torch.from_numpy(X_test).type(torch.float)
    y_train = torch.tensor(y_train)
    y_test = torch.tensor(y_test)
    ## Create train/test datasets
    train_set = custom_dataset(X_train, y_train)
    test_set = custom_dataset(X_test, y_test)
    ## Create train/test loaders
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=32)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=32)
    ## Retrive the correctly ordered labels
    for _type in ['train', 'test']:
        if _type == 'train':
            dataloader = train_loader
        else:
            dataloader = test_loader
        n_sample = 0
        for _, y in dataloader:
            n_sample += torch.sum(torch.isin(y, torch.tensor(studied_class))).item()
        y_true = np.ones(n_sample, dtype='int')
        torch.manual_seed(1)
        count = 0
        for i, (_, target) in enumerate(dataloader):
            target = target[torch.isin(target, torch.tensor(studied_class))]
            batch_size = target.shape[0]
            y_true[count:count + batch_size] = target.cpu().detach().numpy()
            count = count + batch_size
        if _type == 'train':
            y_ori_train = y_true.copy()
        else:
            y_ori_test = y_true.copy()
if name[:3] == "syn":
    studied_class = np.arange(data["n_class"] - 1)
    base_class = data["n_class"] - 1


# Ground truth
counts = {}
genes_per_class = {}
for c in studied_class:
    genes_per_class["C"+str(c)] = []
    counts[c] = 0
    for P in data['useful_paths']["C"+str(c)]:
        for g in data['useful_genes'][P]:
            if g not in genes_per_class["C"+str(c)]:
                genes_per_class["C"+str(c)].append(g)
                counts[c] += 1
    if base_class:
        for P in data['useful_paths']["C"+str(base_class)]:
            for g in data['useful_genes'][P]:
                if g not in genes_per_class["C"+str(c)]:
                    genes_per_class["C"+str(c)].append(g)
                    counts[c] += 1
print(f"{counts} genes need to be retrieved per class.")


# XAI method
XAI_method = "Integrated_Gradients"


# Diffusion
if diffusion:
    D = load_npz(os.path.join(save_path, 'graph', 'diffusion.npz'))


# Important genes defined globally from the test set
chosen_g  = {}

# Test set
attr, y_pred, y_true, labels, features = load_attributions(XAI_method, os.path.join(save_path, save_name), set_name='test')
if diffusion:
    attr = csc_matrix(attr)
    attr = attr.dot(D)
    attr = attr.toarray()
attr = scale_data(attr, _type='norm')

# We only select correctly classified examples.
correct_indices = np.argwhere((y_pred - y_true) == 0)[:, 0]
print("There are {} uncorrect test examples. We remove them from our study.".format(len(y_pred) - len(correct_indices)))

# Retrive sub-groups
if name in ['s1', 's2', 's4']:
    assert (np.sort(np.unique(y_ori_test[y_true==0])) == np.array([0, 1, 2])).all()
    assert (np.sort(np.unique(y_ori_test[y_true==1])) == np.array([3])).all()
    y_true = y_ori_test.copy()
elif name in ['s5',]:
    assert (np.sort(np.unique(y_ori_test[y_true==0])) == np.array([0, 1, 2, 3, 4])).all()
    assert (np.sort(np.unique(y_ori_test[y_true==1])) == np.array([5])).all()
    y_true = y_ori_test.copy()
elif name[:5] == "syn_g":
    y_true = y_ori_test.copy()
attr = attr[correct_indices]
y_true = y_true[correct_indices]

# Ranking
for c in studied_class:
    indices = np.argwhere(y_true == c)[:, 0]
    attr_cls = attr[indices]
    genes = np.argsort(-attr_cls, axis=1)[:, :counts[c]].reshape(-1)
    list_g, nb_g = np.unique(genes, return_counts=True)
    chosen_g["C"+str(c)] = list_g[np.argsort(-nb_g)[:counts[c]]]

# Results
global_score = 0
for c in studied_class:
    global_score += len(set(chosen_g["C"+str(c)]).intersection(set(genes_per_class["C"+str(c)]))) / counts[c]
    print(f'Class {c}  -- common support {np.round(len(set(chosen_g["C"+str(c)]).intersection(set(genes_per_class["C"+str(c)]))) / counts[c] * 100, 2)}')
print(' ')
global_score = global_score / len(studied_class)
print(f"Global ranking score averaged per class on the test set: {np.round(global_score * 100, 2)}")

# Important genes defined locally from the test set
attr, y_pred, y_true, labels, features = load_attributions(XAI_method, os.path.join(save_path, save_name), 'test')
if diffusion:
    attr = csc_matrix(attr)
    attr = attr.dot(D)
    attr = attr.toarray()
attr = scale_data(attr, _type='norm')

# We only select correctly classified examples.
correct_indices = np.argwhere((y_pred - y_true) == 0)[:, 0]
print("There are {} uncorrect test examples. We remove them from our study.".format(len(y_pred) - len(correct_indices)))

# Retrive sub-groups
if name in ['s1', 's2', 's4']:
    assert (np.sort(np.unique(y_ori_test[y_true==0])) == np.array([0, 1, 2])).all()
    assert (np.sort(np.unique(y_ori_test[y_true==1])) == np.array([3])).all()
    y_true = y_ori_test.copy()
elif name in ['s5',]:
    assert (np.sort(np.unique(y_ori_test[y_true==0])) == np.array([0, 1, 2, 3, 4])).all()
    assert (np.sort(np.unique(y_ori_test[y_true==1])) == np.array([5])).all()
    y_true = y_ori_test.copy()
elif name[:5] == "syn_g":
    y_true = y_ori_test.copy()
attr = attr[correct_indices]
y_true = y_true[correct_indices]

# Ranking
ranking = 0
for c in studied_class:
    indices = np.argwhere(y_true == c)[:, 0]
    attr_cls = attr[indices]
    genes = np.argsort(-attr_cls, axis=1)[:, :counts[c]].reshape(-1)
    nb_correct = 0
    for g in genes:
        if g in genes_per_class["C"+str(c)]:
            nb_correct += 1
    ranking += nb_correct / len(genes)
local_score = ranking / len(studied_class)
print(f"Average ranking score on the test examples: {np.round(local_score * 100, 2)}")


# Save
if diffusion:
    file_name = "ranking_diffusion.csv"
else:
    file_name = "ranking.csv"
with open(os.path.join(save_path, save_name, file_name), "w") as f:
    f.write(f"global, {np.round(global_score, 4)}\n")
    f.write(f"local, {np.round(local_score, 4)}\n")
