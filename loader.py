import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler
import os
from dataset import *
from setting import get_setting


### Useful functions
def get_number_features(data):
    for X, y in data:
        n_feat = X.shape[0]
        break
    return n_feat


def get_number_classes(data):
    return len(data.label_key)


def split_indices(data_size, test_size, random_state):
    # Shuffle the indices
    indices = list(range(data_size))
    np.random.seed(random_state)
    np.random.shuffle(indices)
    # Split them
    split = int(np.floor(test_size * data_size))
    train_indices, test_indices = indices[split:], indices[:split]
    # Random seed again
    np.random.seed()
    return train_indices, test_indices


def create_balanced_subset_from_data(data, indices, n):
    """
    Return a subset of the list 'indices' containing 'n' examples for each class. 
    data[i] = (X, y), where y is the class (number between 0 and n_class - 1). 
    """
    # Shuffle the indices to return a different list at each execution.
    # Note that the np.random.shuffle shuffles indices outside this function as well. 
    np.random.shuffle(indices)
    # Select the indices associated with the a class until the number of examples 
    # for this class is reached.
    subset_indices = []
    classes = np.zeros(len(data.label_key))
    for i in indices:
        X, y = data[i]
        classes[y] += 1
        if classes[y] <= n:
            subset_indices.append(i)
    return subset_indices


### Torch loaders
def find_mean_std(data, train_sampler, device, log=True, _sum=False, norm=False):
    loader = torch.utils.data.DataLoader(data, batch_size=len(train_sampler), sampler=train_sampler)
    x, y = next(iter(loader))
    if norm:
        x = x / torch.linalg.norm(x, axis=1).reshape(-1, 1) * 10**5
    if _sum:
        x = 2**x - 1
        x = x / torch.sum(x, dim=1).reshape(-1, 1) * 10**6 
    if log:
        mean, std = torch.log2(x+1).mean(dim=0), torch.log2(x+1).std(dim=0)
    else:
        mean, std = x.mean(dim=0), x.std(dim=0)
    return mean.to(device), std.to(device)


class normalize(nn.Module):
    """Class enabling to 1- Scale the examples. 2 - Center/Reduce the features.
    
    Scaling parameters:
        log  --  all values are log2 transformed
        norm  --  the norm of each sample will be equal to 10**5
        _sum  -- all values are transformed with 2**value - 1. Then, the sum of each sample becomes equal to 10**6.
    """
    def __init__(self, mean, std, log=True, _sum=False, norm=False):
        super().__init__()
        self.mean = mean
        self.std = std
        self.log = log
        self.norm = norm
        self._sum = _sum
        
    def forward(self, x):
        if self.norm:
            x = x / torch.linalg.norm(x, axis=1).reshape(-1, 1) * 10**5
        if self._sum:
            x = 2**x - 1
            x = x / torch.sum(x, dim=1).reshape(-1, 1) * 10**6
        if self.log:
            return (torch.log2(x+1) - self.mean) / self.std
        else:
            return (x - self.mean) / self.std


def load_dataloader(data_path, name, device, batch_size):
    if name in ["pancan", "BRCA", "KIRC", "COAD", "LUAD"]:
        # Load data
        database, label_name, log, _sum = get_setting(name)
        data = TCGA_dataset(data_path, database, name, label_name)

        # Information
        n_class = get_number_classes(data)
        n_feat = get_number_features(data)
        class_name = data.label_key
        feat_name = data.genes_IDs
        n_sample = len(data)
        
        # Create train/test loaders
        random_state = 43
        test_size = 0.4
        train_indices, test_indices = split_indices(n_sample, test_size, random_state)
        train_sampler = SubsetRandomSampler(train_indices)
        train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, sampler=train_sampler)
        test_sampler = SubsetRandomSampler(test_indices)
        test_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, sampler=test_sampler)

        # Normalization
        mean, std = find_mean_std(data, train_sampler, device, log, _sum)
        if name != 'pancan':
            std = torch.ones(std.shape).to(device)
        transform = normalize(mean, std, log, _sum)
        
    else:
        # Load data
        data = np.load(os.path.join(data_path, f'{name}.npy'), allow_pickle=True).item()
        X = data['X']
        y = data['y']
        
        # Information
        n_class = data['n_class']
        if name in ['SimuA', 'SimuB', 'SimuC'] or name[:3] == "syn":
            n_class = 2
        n_feat = data['n_gene']
        class_name = np.arange(n_class)
        feat_name = None
        n_sample = len(X)
        
        # Create train/test sets
        random_state = 43
        test_size = 0.4
        train_indices, test_indices = split_indices(n_sample, test_size, random_state)
        X_train, X_test, y_train, y_test = split_data_from_indices(X, y, train_indices, test_indices)
        X_train = torch.from_numpy(X_train).type(torch.float)
        X_test = torch.from_numpy(X_test).type(torch.float)
        
        # Relabel the classes to create an heterogeneous class
        if name in ['SimuB', 'SimuC'] or name[:5] == "syn_g":
            for c in range(1, data['n_class']-1):
                y_train[np.argwhere(y_train == c)] = np.zeros(np.argwhere(y_train == c).shape)
                y_test[np.argwhere(y_test == c)] = np.zeros(np.argwhere(y_test == c).shape)
            y_train[np.argwhere(y_train == data['n_class']-1)] = np.ones(np.argwhere(y_train == data['n_class']-1).shape)
            y_test[np.argwhere(y_test == data['n_class']-1)] = np.ones(np.argwhere(y_test == data['n_class']-1).shape)

        # Create train/test datasets
        y_train = torch.tensor(y_train)
        y_test = torch.tensor(y_test)
        for _class in range(n_class):
            assert torch.sum(y_train==_class) >= 10
        train_set = custom_dataset(X_train, y_train)
        test_set = custom_dataset(X_test, y_test)
        
        # Create train/test loaders
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)
        
        # Normalization
        mean, std = find_mean_std(train_set, np.arange(len(train_set)), device, log=False, _sum=False)
        transform = normalize(mean, std, log=False, _sum=False)
        
    return train_loader, test_loader, n_class, n_feat, class_name, feat_name, transform, n_sample



### Datasets
def load_dataset(data_path, name, normalize=False):
    if name in ["pancan", "BRCA", "KIRC", "COAD", "LUAD"]:
        # Load data
        database, label_name, log, _sum = get_setting(name)
        data = TCGA_dataset(data_path, database, name, label_name)
        if name != "pancan":
            std = False
        else:
            std = True
        # Create train/test sets
        X_train, X_test, y_train, y_test = create_train_test(data, test_size=0.4, random_state=43, normalize=normalize, log=log, _sum=_sum, std=std)
        X_train = X_train.numpy()
        X_test = X_test.numpy()
        y_train = np.ravel(y_train.numpy())
        y_test = np.ravel(y_test.numpy())
        # Information
        n_class = get_number_classes(data)
        n_feat = get_number_features(data)
        class_name = list(data.label_map.keys())

        # Information
        n_class = get_number_classes(data)
        n_feat = get_number_features(data)
        class_name = list(data.label_map.keys())
    elif name in ['SIMU1', 'SIMU2', 'SimuA', 'SimuB', 'SimuC'] or name[:3] == "syn":
        # Load data
        data = np.load(os.path.join(data_path, f'{name}.npy'), allow_pickle=True).item()
        X = data['X']
        y = data['y']
        # Create train/test sets
        train_indices, test_indices = split_indices(len(X), test_size=0.4, random_state=43)
        X_train, X_test, y_train, y_test = split_data_from_indices(X, y, train_indices, test_indices)
        if normalize:
            X_train, X_test = normalize_train_test(X_train, X_test, log=False, _sum=False, std=True) 
        # Information
        n_class = data['n_class']
        n_feat = data['n_gene']
        print(f"In our dataset, we have {n_class} classes. Each example contains {n_feat} features.")
        class_name = np.arange(n_class)
    return X_train, X_test, y_train, y_test, n_class, n_feat, class_name


def create_train_test(data, test_size, random_state, normalize=True, classes=None, log=True, _sum=False, std=True):
    # Get training and test set
    X, y = get_X_y(data)
    train_indices, test_indices = split_indices(len(data), test_size, random_state)
    X_train, X_test, y_train, y_test = split_data_from_indices(X, y, train_indices, test_indices)
    # Select a subset of classes
    if classes is not None:
        train_indices = [item for _class in classes for item in torch.argwhere(y_train.reshape(-1) == _class)[:, 0].cpu().numpy()]
        X_train = X_train[train_indices]
        y_train = y_train[train_indices]
        test_indices = [item for _class in classes for item in torch.argwhere(y_test.reshape(-1) == _class)[:, 0].cpu().numpy()]
        X_test = X_test[test_indices]
        y_test = y_test[test_indices]
    # Normalize them
    if normalize:
        X_train, X_test = normalize_train_test(X_train, X_test, log, _sum, std)
    return X_train, X_test, y_train, y_test


def get_X_y(data):
    """Load the entire dataset into two tensors."""
    data_size = len(data)
    n_feat = get_number_features(data)

    X = torch.zeros((data_size, n_feat))
    y = torch.zeros((data_size, 1))
    for i, (sample, label) in enumerate(data):
        X[i] += sample
        y[i] += label

    return X, y


def split_data_from_indices(X, y, train_indices, test_indices):
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


def normalize_train_test(X_train, X_test, log=True, _sum=False, std=True):
    if str(X_train.dtype).split('.')[0] == 'torch':
        if _sum:
            X_train = 2**X_train - 1
            X_train = X_train / torch.sum(X_train, dim=1).reshape(-1, 1) * 10**6
            X_test = 2**X_test - 1
            X_test = X_test / torch.sum(X_test, dim=1).reshape(-1, 1) * 10**6 
        if log:
            X_train = torch.log2(X_train+1)
            X_test = torch.log2(X_test+1)
        mean = X_train.mean(dim=0)
        if std:
            std = X_train.std(dim=0)
    else:
        if _sum:
            X_train = 2**X_train - 1
            X_train = X_train / np.sum(X_train, axis=1).reshape(-1, 1) * 10**6
            X_test = 2**X_test - 1
            X_test = X_test / np.sum(X_test, axis=1).reshape(-1, 1) * 10**6 
        if log:
            X_train = np.log2(X_train+1)
            X_test = np.log2(X_test+1)
        mean = np.mean(X_train, axis=0)
        if std:
            std = np.std(X_train, axis=0)
    if std is not False:
        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std
    else:
        X_train = (X_train - mean)
        X_test = (X_test - mean)
    return X_train, X_test


def scale_data(X, _type='sum', factor=10**6):
    """X is a matrix of shape [n, p] containing n vectors. 
    Return a matrix of shape [n, p].
    If _type='sum', the sum of X[k, :] is 'factor'.
    If _type='norm', the Euclidean norm of X[k, :] is set to 1.
    """
    if _type == 'sum':
        return X / np.reshape(np.sum(X, axis=1), (-1, 1)) * factor
    elif _type == 'norm':
        return X / np.reshape(np.linalg.norm(X, axis=1), (-1, 1))


def transform_data(X, name):
    assert name in ['log2', 'sqrt', 'reduce_center', 'pearson_regularization'], "name should be 'log2', 'sqrt', 'reduce_center', 'pearson_regularization'"
    if name == 'log2':
        return np.log2(X + 1)
    elif name == 'sqrt':
        return np.sqrt(X+1)
    elif name == 'reduce_center':
        mean, std = np.mean(X, axis=0), np.std(X, axis=0)
        return (X - mean) / std
    elif name == 'pearson_regularization':
        mean = np.mean(X, axis=0)
        return X / np.sqrt(mean+1)
