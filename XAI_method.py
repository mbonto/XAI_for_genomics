import torch
from captum.attr import IntegratedGradients, KernelShap
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
    
    
def compute_attributes_from_a_dataloader(model, dataloader, transform, device, method="Integrated_Gradients", n_steps=100, n_samples=10, baselines=None):
    """
    Return importance scores attributed by an XAI method to each feature of each input.
    
    Parameters:
    model  -- neural network
    X  --  inputs, torch tensor (n_sample, n_feat)
    y  --  labels, torch tensor (n_sample)
    method -- "Integrated_Gradients", "Kernel_Shap"
    n_steps  --  integer, used if method == "Integrated_Gradients"
    n_samples  --  integer, used if method == "Kernel_Shap"
    baselines  --  input, torch tensor (1, n_feat) used a reference input in "Integrated_Gradients"
    """
    # Informations
    n_sample = len(dataloader.sampler)
    x, _ =  next(iter(dataloader))
    n_feat = x.shape[1]
    
    # Compute the attributions
    attr = torch.zeros(n_sample, n_feat).to(device)
    y_pred = np.ones(n_sample)
    y_true = np.ones(n_sample, dtype='int')
    if method == "Integrated_Gradients":
        xai = IntegratedGradients(model)
    elif method == "Kernel_Shap":
        xai = KernelShap(model)   
    torch.manual_seed(1)
    count = 0
    for i, (x, target) in enumerate(dataloader):
        print(i, end='\r')
        batch_size = x.shape[0]
        x = x.to(device)
        if transform:
            x = transform(x)
        target = target.to(device)
        if method == "Integrated_Gradients":
            attr[count:count + batch_size, :] = xai.attribute(x, target=target, n_steps=n_steps, baselines=baselines, internal_batch_size=batch_size)
        elif method == "Kernel_Shap":
            attr[count:count + batch_size, :] = xai.attribute(x, target=target, n_samples=n_samples)
        outputs = model(x)
        _, pred = torch.max(outputs.data, 1)
        y_true[count:count + batch_size] = target.cpu().detach().numpy()
        y_pred[count:count + batch_size] = pred.cpu().detach().numpy()
        count = count + batch_size
    attr = attr.detach().cpu().numpy()
    return attr, y_true, y_pred


def check_ig_from_a_dataloader(attr, model, dataloader, transform, device, baseline, save_name=None, show=True):
    """
    For each input, we should have `sum of the attributions = model(input) - model(baseline)`.
    If this is not the case, increase the number of steps and recompute the attributions.
    
    Parameters:
    attr  --  attributions, torch tensor (batch_size, n_feat)
    model  -- neural network
    X  --  inputs, torch tensor (batch_size, n_feat)
    y  --  labels, torch tensor (batch_size)
    baseline  -- torch tensor (1, n_feat)
    """
    _sum = np.round(np.sum(attr, axis=1) * 100, decimals=2)
    
    n_sample = len(dataloader.sampler)
    output_X = np.zeros(n_sample)
    output_baseline = np.zeros(n_sample)
    
    torch.manual_seed(1)
    count = 0
    for i, (x, target) in enumerate(dataloader):
        print(i, end='\r')
        batch_size = x.shape[0]
        
        x = x.to(device)
        
        if transform:
            x = transform(x)
            
        target = target.to(device)
        
        output_X[count:count + batch_size] = torch.take_along_dim(model(x), target.reshape(-1, 1), dim=1).reshape(-1).cpu().detach().numpy()
        output_baseline[count:count + batch_size] = torch.take_along_dim(model(baseline).repeat(x.shape[0], 1), target.reshape(-1, 1), dim=1).reshape(-1).cpu().detach().numpy()
        count = count + batch_size
    
    diff = np.round((output_X - output_baseline) * 100, 2)
    
    sns.violinplot(x=_sum-diff)
    plt.xlabel("sum of the attributions - (model(input) - model(baseline))", labelpad=50)
        
    if save_name:
        plt.savefig(save_name, bbox_inches='tight', dpi=150)
    
    if show:
        plt.show()
    plt.close('all')
    
    return np.max(np.abs(_sum-diff))
    
    
def load_attributions(XAI_method, save_path, set_name):
    checkpoint = torch.load(os.path.join(save_path, '{}_{}.pt'.format(XAI_method, set_name)))
    return checkpoint['features_score'], checkpoint['predictions'], checkpoint['true_labels'], checkpoint['labels_name'], checkpoint['features_name']



def save_attributions(features_score, features_name, model, XAI_method, predictions, true_labels, labels_name, save_path, set_name):
    torch.save({'features_score': features_score,
            'predictions': predictions,
            'true_labels': true_labels,
            'labels_name': labels_name,
            'features_name': features_name,
            'variables': model.variables,
            'name': model.name,
            }, os.path.join(save_path, "{}_{}.pt".format(XAI_method, set_name)))
