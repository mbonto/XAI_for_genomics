# Librairies
import os
import sys
code_path = os.path.split(os.path.split(os.getcwd())[0])[0]
sys.path.append(code_path)
import numpy as np
import torch
import argparse
from setting import *
from utils import *
from loader import *
from evaluate import *
from models import *
from XAI_method import *
set_pyplot()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Arguments
argParser = argparse.ArgumentParser()
argParser.add_argument("-n", "--name", type=str, help="dataset name")
argParser.add_argument("-m", "--model", type=str, help="model name (LR, MLP, DiffuseLR, DiffuseMLP)")
argParser.add_argument("-s", "--step", type=int, help="number of steps", default=3000)
argParser.add_argument("--set", type=str, help="set (train or test)")
argParser.add_argument("--exp", type=int, help="experiment number", default=1)
args = argParser.parse_args()
name = args.name
model_name = args.model
n_step = args.step
set_name = args.set
exp = args.exp
print('Model    ', model_name)


# Path
save_path = get_save_path(name, code_path)
data_path = get_data_path(name)
save_name = os.path.join(model_name, f"exp_{exp}")

# Dataset
train_loader, test_loader, n_class, n_feat, class_name, feat_name, transform, n_sample = load_dataloader(data_path, name, device, batch_size=32)


# Set
if set_name == 'train':
    loader = train_loader
elif set_name == 'test':
    loader = test_loader
    
    
# Model
softmax = True
n_layer, n_hidden_feat = get_hyperparameters(name, model_name)
model = load_model(model_name, n_feat, n_class, softmax, device, save_path, n_layer, n_hidden_feat)
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
print(f"The output of the baseline is {model(baseline)}")

    
# Integrated gradients
valid = False
while not valid:
    XAI_method = "Integrated_Gradients"
    attr, y_true, y_pred = compute_attributes_from_a_dataloader(model, loader, transform, device, studied_class, XAI_method, n_step, baseline=baseline)
    # With Integrated_Gradients, for each input, the sum of the attributions should be equal to model(input) - model(baseline).
    # Otherwise, increase the number of steps (n_steps).
    save = os.path.join(save_path, save_name, "Figures", f"IG_check_{set_name}.png")
    ## indices = np.argwhere(y_true == y_pred)
    score = check_ig_from_a_dataloader(attr, model, loader, transform, device, baseline, studied_class, save, show=False)
    
    if score > 1:
        n_step = n_step + 5000
        print(f"Maximal gap: {np.round(score, 6)}. There is at least one sample for which the difference between the sum of the attributions and the predictions is higher than 1. Run again IG with {n_step} steps.")
    else:
        valid = True
        print(f"Maximal gap: {np.round(score, 6)}.")
    

# Save
save_attributions(attr, None, model, XAI_method, y_pred, y_true, np.arange(n_class), os.path.join(save_path, save_name), set_name)
print(' ')
