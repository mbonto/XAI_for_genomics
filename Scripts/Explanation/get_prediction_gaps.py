# Libraries
import os
import sys
code_path = os.path.split(os.path.split(os.getcwd())[0])[0]
sys.path.append(code_path)

import numpy as np
import torch
import argparse

from setting import *
from dataset import *
from loader import *
from evaluate import *
from models import *
from XAI_method import *
from XAI_interpret import *
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Arguments
argParser = argparse.ArgumentParser()
argParser.add_argument("-n", "--name", type=str, help="dataset name")
argParser.add_argument("-m", "--model", type=str, help="model name (LR, MLP, DiffuseLR, DiffuseMLP)")
argParser.add_argument("--set", type=str, help="set (train or test)")
argParser.add_argument("--gap", type=int, help="prediction gaps are computed every `gap` features removed", default=10)
argParser.add_argument("--exp", type=int, help="experiment number", default=1)
args = argParser.parse_args()
name = args.name
model_name = args.model
set_name = args.set
gap = args.gap
exp = args.exp
print('Model    ', model_name)


# Path
save_path = get_save_path(name, code_path)
data_path = get_data_path(name)
save_name = os.path.join(model_name, f"exp_{exp}")


# Load a dataset
train_loader, test_loader, n_class, n_feat, class_name, feat_name, transform, n_sample = load_dataloader(data_path, name, device, batch_size=32)


# Load a model
softmax = True
n_layer, n_hidden_feat = get_hyperparameters(name, model_name)
model = load_model(model_name, n_feat, n_class, softmax, device, save_path, n_layer, n_hidden_feat)

# Parameters
checkpoint = torch.load(os.path.join(save_path, save_name, 'checkpoint.pt'))
model.load_state_dict(checkpoint['state_dict'])
model.eval()


# Assert that the model and the data are coherent
assert compute_accuracy_from_model_with_dataloader(model, train_loader, transform, device) == checkpoint['train_acc']
assert compute_accuracy_from_model_with_dataloader(model, test_loader, transform, device) == checkpoint['test_acc']


# Baseline
if name in ['BRCA', 'KIRC', 'SimuA', 'SimuB', 'SimuC']:
    base_class = 1
    studied_class = [0,]
else:
    base_class = None
    studied_class = list(np.arange(n_class))
baseline = get_baseline(train_loader, device, n_feat, transform, base_class)
default_output = model(baseline).detach().cpu().numpy()


# Data of interest
if set_name == 'train':
    loader = train_loader
elif set_name == 'test':
    loader = test_loader
    

# Load the attribution scores
XAI_method = "Integrated_Gradients"
attr, y_pred, y_true, labels, features = load_attributions(XAI_method, os.path.join(save_path, save_name), set_name=set_name)

# Normalize them
attr = scale_data(attr, _type='norm')


# Local prediction gap ...
print("Local PGs")

# ... on unimportant features
PGU = prediction_gap_with_dataloader(model, loader, transform, attr, gap, baseline, studied_class, None, "unimportant", y_true, y_pred)
print('Average PGU', np.round(np.mean(list(PGU.values())), 4) * 100)
adj_PGU = {}
for c in studied_class:
    adj_PGU[c] = PGU[c] / (1 - default_output[0, c])
print('Adjusted average PGU', np.round(np.mean(list(adj_PGU.values())), 4) * 100)

# ... on important features
PGI = prediction_gap_with_dataloader(model, loader, transform, attr, gap, baseline, studied_class, None, "important", y_true, y_pred)
print('Average PGI', np.round(np.mean(list(PGI.values())), 4) * 100)

# Save
with open(os.path.join(save_path, save_name, "local_XAI.csv"), "w") as f:
    f.write(f"PGU, {np.round(np.mean(list(PGU.values())), 4)}\n")
    f.write(f"PGU_adjusted, {np.round(np.mean(list(adj_PGU.values())), 4)}\n")
    f.write(f"PGI, {np.round(np.mean(list(PGI.values())), 4)}\n")
    
    
# Global prediction gap ...
print('\nGlobal PGs')
# ... on unimportant features
indices = get_features_order(attr, _type="increasing")
PGU = prediction_gap_with_dataloader(model, loader, transform, attr, gap, baseline, studied_class, indices, None, y_true, y_pred)
print('Average PGU', np.round(np.mean(list(PGU.values())), 4) * 100)
adj_PGU = {}
for c in studied_class:
    adj_PGU[c] = PGU[c] / (1 - default_output[0, c])
print('Adjusted average PGU', np.round(np.mean(list(adj_PGU.values())), 4) * 100)

# Prediction gap on important features
indices = get_features_order(attr, _type="decreasing")
PGI = prediction_gap_with_dataloader(model, loader, transform, attr, gap, baseline, studied_class, indices, None, y_true, y_pred)
print('Average PGI', np.round(np.mean(list(PGI.values())), 4) * 100)


# Prediction gap on random features
PGRs = []
for t in range(30):
    indices = get_features_order(attr, _type="random")
    PGR = prediction_gap_with_dataloader(model, loader, transform, attr, gap, baseline, studied_class, indices, None, y_true, y_pred)
    PGRs.append(np.round(np.mean(list(PGR.values())), 4))
print('Average PGR', np.round(np.mean(PGRs), 4) * 100)


# Save
with open(os.path.join(save_path, save_name, "global_XAI.csv"), "w") as f:
    f.write(f"PGU, {np.round(np.mean(list(PGU.values())), 4)}\n")
    f.write(f"PGU_adjusted, {np.round(np.mean(list(adj_PGU.values())), 4)}\n")
    f.write(f"PGR, {np.round(np.mean(PGRs), 4)}\n")
    f.write(f"PGI, {np.round(np.mean(list(PGI.values())), 4)}\n")
    f.write(f"list_PGR, {PGRs}\n")
