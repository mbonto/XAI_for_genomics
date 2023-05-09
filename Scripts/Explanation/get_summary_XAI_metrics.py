# Librairies
import os
import sys
code_path = os.path.split(os.path.split(os.getcwd())[0])[0]
sys.path.append(code_path)
import argparse
import numpy as np
from setting import *

# Arguments
argParser = argparse.ArgumentParser()
argParser.add_argument("-n", "--name", type=str, help="dataset name")
argParser.add_argument("-m", "--model", type=str, help="model name (LR, MLP, DiffuseLR, DiffuseMLP)")
argParser.add_argument("--n_repet", type=int, help="Results are averaged for all experiments between 1 and `n_repet`")

args = argParser.parse_args()
name = args.name
model_name = args.model
n_repet = args.n_repet
print('Model    ', model_name)

# Path
save_path = get_save_path(name, code_path)
data_path = get_data_path(name)

# Summarize local PGs
exps = np.arange(1, n_repet)
PGU = []
PGI = []

for exp in exps:
    save_name = os.path.join(model_name, f"exp_{exp}") 
    with open(os.path.join(save_path, save_name, "local_XAI.csv"), "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split(', ')
            if line[0] == 'PGU':
                PGU.append(float(line[1]) * 100)
            if line[0] == 'PGI':
                PGI.append(float(line[1]) * 100)
assert len(PGU) == len(exps)
print("Local Prediction Gaps")
print(f"PGU with {model_name} on {name}: {np.round(np.mean(PGU) , 2)} +- {np.round(np.std(PGU) , 2)}")
print(f"PGI with {model_name} on {name}: {np.round(np.mean(PGI) , 2)} +- {np.round(np.std(PGI) , 2)}")


# Summarize global PGs
PGU = []
PGI = []
PGR = []

for exp in exps:
    save_name = os.path.join(model_name, f"exp_{exp}")   
    with open(os.path.join(save_path, save_name, "global_XAI.csv"), "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split(', ')
            if line[0] == 'PGU':
                PGU.append(float(line[1]) * 100)
            if line[0] == 'PGI':
                PGI.append(float(line[1]) * 100)
            if line[0] == 'PGR':
                PGR.append(float(line[1]) * 100)
assert len(PGU) == len(exps)
print(' ')
print("Global Prediction Gaps")
print(f"PGU with {model_name} on {name}: {np.round(np.mean(PGU) , 2)} +- {np.round(np.std(PGU) , 2)}")
print(f"PGI with {model_name} on {name}: {np.round(np.mean(PGI) , 2)} +- {np.round(np.std(PGI) , 2)}")
print(f"PGR with {model_name} on {name}: {np.round(np.mean(PGR) , 2)} +- {np.round(np.std(PGR) , 2)}")


# Summarize FA
if name in ['SimuA', 'SimuB', 'SimuC', 'SIMU1', 'SIMU2']:
    local = []
    _global = []
    
    for exp in exps:
        save_name = os.path.join(model_name, f"exp_{exp}")   
        with open(os.path.join(save_path, save_name, "ranking.csv"), "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(', ')
                if line[0] == 'local':
                    local.append(float(line[1]) * 100)
                if line[0] == 'global':
                    _global.append(float(line[1]) * 100)
    assert len(local) == len(exps)
    print(f"Local FA with {model_name} on {name}: {np.round(np.mean(local) , 2)} +- {np.round(np.std(local) , 2)}")
    print(f"Global FA with {model_name} on {name}: {np.round(np.mean(_global) , 2)} +- {np.round(np.std(_global) , 2)}")

