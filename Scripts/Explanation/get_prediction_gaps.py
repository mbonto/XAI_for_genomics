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
args = argParser.parse_args()
name = args.name
model_name = args.model
set_name = args.set
gap = args.gap
print('Model    ', model_name)


# Path
save_path = get_save_path(name, code_path)
data_path = get_data_path(name)


# Load a dataset
train_loader, test_loader, n_class, n_feat, class_name, feat_name, transform, n_sample = load_dataloader(data_path, name, device, batch_size=32)


# Load a model
softmax = True
model = load_model(model_name, n_feat, n_class, softmax, device, save_path)
## Parameters
checkpoint = torch.load(os.path.join(save_path, model_name, 'checkpoint.pt'))
model.load_state_dict(checkpoint['state_dict'])
model.eval()


# Assert that the model and the data are coherent
assert compute_accuracy_from_model_with_dataloader(model, train_loader, transform, device) == checkpoint['train_acc']
assert compute_accuracy_from_model_with_dataloader(model, test_loader, transform, device) == checkpoint['test_acc']


# Baseline
baseline = torch.zeros(1, n_feat).to(device)
default_output = model(baseline).detach().cpu().numpy()


# Data of interest
if set_name == 'train':
    loader = train_loader
elif set_name == 'test':
    loader = test_loader
    

# Load the attribution scores
XAI_method = "Integrated_Gradients"
attr, y_pred, y_true, labels, features = load_attributions(XAI_method, os.path.join(save_path, model_name), set_name=set_name)
## Normalize them
attr = scale_data(attr, _type='norm')


# Prediction gap on unimportant features
_type = 'unimportant'
PGU = prediction_gap_with_dataloader(model, loader, transform, attr, n_class, _type, y_true, y_pred, gap)
print('Average PGU', np.round(np.mean(PGU), 4))
adj_PGU = PGU / (1 - default_output)
print('Adjusted average PGU', np.round(np.mean(adj_PGU), 4))


# Prediction gap on important features
_type = 'important'
PGI = prediction_gap_with_dataloader(model, loader, transform, attr, n_class, _type, y_true, y_pred, gap)
print('Average PGI', np.round(np.mean(PGI), 4))
