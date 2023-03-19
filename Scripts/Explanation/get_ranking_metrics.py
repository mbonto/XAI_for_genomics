# Libraries
import os
import sys
code_path = os.path.split(os.path.split(os.getcwd())[0])[0]
sys.path.append(code_path)

import numpy as np
import matplotlib.pyplot as plt
import torch
import argparse

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
args = argParser.parse_args()
name = args.name
model_name = args.model
print('Model    ', model_name)


# Path
save_path = get_save_path(name, code_path)
data_path = get_data_path(name)


# Dataset
data = np.load(os.path.join(data_path, f'{name}.npy'), allow_pickle=True).item()
n_class = data['n_class']


# Adjacency matrix
A = np.load(os.path.join(save_path, 'graph', 'pearson_correlation.npy'), allow_pickle=True)
A = get_normalized_adjaceny_matrix(A, 0.)


# Ground truth
counts = np.zeros(n_class, dtype='int')
genes_per_class = {}
for c in range(n_class):
    genes_per_class["C"+str(c)] = []
    for P in data['useful_paths']["C"+str(c)]:
        for g in data['useful_genes'][P]:
            if g not in genes_per_class["C"+str(c)]:
                genes_per_class["C"+str(c)].append(g)
                counts[c] += 1
print(f"{counts} genes need to be retrieved per class.")


# XAI method
XAI_method = "Integrated_Gradients"


# Important genes defined globally from the training set
chosen_g_wo_diff  = {}
chosen_g_w_diff = {}
attr, y_pred, y_true, labels, features = load_attributions(XAI_method, os.path.join(save_path, model_name), set_name='train')
attr = scale_data(attr, _type='norm')
## We only select correctly classified examples.
correct_indices = np.argwhere((y_pred - y_true) == 0)[:, 0]
print("There are {} uncorrect train examples. We remove them from our study.".format(len(y_pred) - len(correct_indices)))
attr = attr[correct_indices]
y_true = y_true[correct_indices]
## Ranking
for c in range(n_class):
    indices = np.argwhere(y_true == c)[:, 0]
    attr_cls = attr[indices]
    ## Without diffusion
    genes = np.argsort(-attr_cls, axis=1)[:, :counts[c]].reshape(-1)
    list_g, nb_g = np.unique(genes, return_counts=True)
    chosen_g_wo_diff["C"+str(c)] = list_g[np.argsort(-nb_g)[:counts[c]]]
    ## With diffusion
    new_attr = np.matmul(attr_cls, A)
    genes = np.argsort(-new_attr, axis=1)[:, :counts[c]].reshape(-1)
    list_g, nb_g = np.unique(genes, return_counts=True)
    chosen_g_w_diff["C"+str(c)] = list_g[np.argsort(-nb_g)[:counts[c]]]
## Results
for c in range(n_class):
    print(f'Class {c}  -- common support {np.round(len(set(chosen_g_wo_diff["C"+str(c)]).intersection(set(genes_per_class["C"+str(c)]))) / counts[c], 2)},  + diffusion {np.round(len(set(chosen_g_w_diff["C"+str(c)]).intersection(set(genes_per_class["C"+str(c)]))) / counts[c], 2)}')
print(' ')


# Important genes defined locally from the test set
attr, y_pred, y_true, labels, features = load_attributions(XAI_method, os.path.join(save_path, model_name), 'test')
attr = scale_data(attr, _type='norm')
## We only select correctly classified examples.
correct_indices = np.argwhere((y_pred - y_true) == 0)[:, 0]
print("There are {} uncorrect test examples. We remove them from our study.".format(len(y_pred) - len(correct_indices)))
attr = attr[correct_indices]
y_true = y_true[correct_indices]
## Ranking
ranking = 0
diffused_ranking = 0
for c in range(n_class):
    indices = np.argwhere(y_true == c)[:, 0]
    attr_cls = attr[indices]
    ## Without diffusion
    genes = np.argsort(-attr_cls, axis=1)[:, :counts[c]].reshape(-1)
    nb_correct = 0
    for g in genes:
        if g in genes_per_class["C"+str(c)]:
            nb_correct += 1
    ranking += nb_correct / len(genes)
    ## With diffusion
    new_attr = np.matmul(attr_cls, A)
    genes = np.argsort(-new_attr, axis=1)[:, :counts[c]].reshape(-1)
    nb_correct = 0
    for g in genes:
        if g in genes_per_class["C"+str(c)]:
            nb_correct += 1
    diffused_ranking += nb_correct / len(genes)
print(f"Average ranking score on the test examples: {np.round(ranking / n_class, 2)}")
print(f"+ diffusion: {np.round(diffused_ranking / n_class, 2)}")

