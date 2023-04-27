# Libraries
import os
import sys
import numpy as np
from scipy.sparse import csc_matrix, eye, save_npz
code_path = os.path.split(os.path.split(os.getcwd())[0])[0]
sys.path.append(code_path)
import argparse
from setting import *
from loader import *
from graphs import *
from utils import *


# Arguments
argParser = argparse.ArgumentParser()
argParser.add_argument("--name", type=str, help="dataset name")
args = argParser.parse_args()
name = args.name


# Path
data_path = get_data_path(name)
save_path = get_save_path(name, code_path)
create_new_folder(os.path.join(save_path, "graph"))


# Load dataset
X_train, X_test, y_train, y_test, n_class, n_feat, class_name = load_dataset(data_path, name)


# Infer the graph
A = get_a_graph(X_train, method='pearson_correlation')
remove_diag(A)
if name in ["pancan", "SIMU1", "SIMU2"]:
    np.save(os.path.join(save_path, "graph", "pearson_correlation.npy"), A)


# Sparse version
t = 0.9
A = (A > t) * A
A = csc_matrix(A)


# Diffusion version
D = get_normalized_adjaceny_matrix(A)
save_npz(os.path.join(save_path, 'graph', f'diffusion'), A)