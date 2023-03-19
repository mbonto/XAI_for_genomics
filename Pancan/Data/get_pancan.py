# Libraries
import os
import sys
code_path = os.path.split(os.path.split(os.getcwd())[0])[0]
sys.path.append(code_path)
from setting import data_path
from download_data import *


# Download
data_path = os.path.join(data_path, 'tcga')
database = 'pancan'
cancer = 'pancan'
download_dataset(data_path, database, cancer)