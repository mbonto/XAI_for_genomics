# Libraries
import os
import sys
import numpy as np
code_path = os.path.split(os.path.split(os.getcwd())[0])[0]
sys.path.append(code_path)
from setting import *
from loader import *
from graphs import *
from utils import *


# Path
name = 'pancan'
save_path = get_save_path(name, code_path)
data_path = get_data_path(name)


# Load dataset
X_train, X_test, y_train, y_test, n_class, n_feat, class_name = load_dataset(data_path, name)


# Infer the graph
A = get_a_graph(X_train, method='pearson_correlation')
remove_diag(A)
create_new_folder(os.path.join(save_path, "graph"))
np.save(os.path.join(save_path, "graph", "pearson_correlation.npy"), A)