# Librairies
import os
import sys
code_path = os.path.split(os.path.split(os.getcwd())[0])[0]
sys.path.append(code_path)
import numpy as np
import argparse
from setting import *
from loader import *
from XAI_method import *
from XAI_interpret import *


# Arguments
argParser = argparse.ArgumentParser()
argParser.add_argument("-n", "--name", type=str, help="dataset name")
argParser.add_argument("-m", "--model", type=str, help="model name (LR, MLP, DiffuseLR, DiffuseMLP)")
argParser.add_argument("--set", type=str, help="set (train, test)")
args = argParser.parse_args()
name = args.name
model_name = args.model
set_name = args.set
print('Model    ', model_name)


# Path
save_path = get_save_path(name, code_path)


# Attributions
XAI_method = "Integrated_Gradients"
attr, y_pred, y_true, labels, features = load_attributions(XAI_method, os.path.join(save_path, model_name), set_name=set_name)
correct_indices = np.argwhere((y_pred - y_true) == 0)[:, 0]
print("There are {} uncorrect examples. We remove them from our study.".format(len(y_pred) - len(correct_indices)))
attr = attr[correct_indices]
y_true = y_true[correct_indices]
y_pred = y_pred[correct_indices]
## Normalize
attr = scale_data(attr, _type='norm')


# Attributions averaged per class
method = 'mean'
## Compute the relevance scores averaged per feature and per class.
classes = np.arange(0, len(labels))
scores = {}
for _class in classes:
    cls_attr = get_attributions_per_class(attr, y_true, _class, method)
    cls_sorted_indices = sort_attribution_scores(cls_attr)
    scores[_class] = {'attr': cls_attr, 'sorted_indices': cls_sorted_indices}
## Same thing without distinguishing the classes.
avg_attr = normalize_attribution_scores(attr, method)
avg_sorted_indices = sort_attribution_scores(avg_attr)
scores["general"] = {'attr': avg_attr, 'sorted_indices': avg_sorted_indices}
## Save
np.save(os.path.join(save_path, model_name, "{}_scores_with_{}_{}.npy".format(XAI_method, method, set_name)), scores)



