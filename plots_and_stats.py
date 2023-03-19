import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import colorcet as cc
from loader import get_X_y


def describe_dataset(data):
    # Number of classes
    print(f'The dataset contains {len(data.label_key)} classes. Here are the classes and their respective number of samples:')
    # Number of samples per class
    classes = np.zeros(len(data.label_key))
    for X, y in data:
        classes[y] += 1
    for label in data.label_key:
        index = data.label_map[label]
        print(f'\t{label}: {int(classes[index])}')
    # Number of features
    for X, y in data:
        n_feat = X.shape[0]
        break
    print(f'In total, there are {len(data)} samples, each of them containing {n_feat} features.')


def describe_subset_of_dataset(data, indices):
    # Number of samples per class
    print(f'There are {len(indices)} samples. Here are the classes and their respective number of samples:')
    classes = np.zeros(len(data.label_key))
    for i in indices:
        X, y = data[i]
        classes[y] += 1
    for label in data.label_key:
        index = data.label_map[label]
        print(f'\t{label}: {int(classes[index])}')


def get_min_samples_per_class(data, indices):
    classes = np.zeros(len(data.label_key))
    for i in indices:
        X, y = data[i]
        classes[y] += 1
    return int(min(classes))
        
        
def describe_dataloader(data_loader):
    # Number of samples per class
    classes = np.zeros(len(data_loader.dataset.label_key))
    for X, y in data_loader:
        classes[y] += 1
    for label in data.label_key:
        index = data.label_map[label]
        print(f'\t{label}: {int(classes[index])}')
    # Number of features
    for X, y in data_loader:
        n_feat = X.shape[1]
        break
    print(f'In total, there are {int(np.sum(classes))} samples, each of them containing {n_feat} features.')



def plot_random_gene_expression(data, group_by_classes=True, gene_index=None, log_scale=False, log=False, unit='count'):
    info_X, info_y = get_X_y(data)    
    info_X = np.array(info_X)
    info_y = np.ravel(np.array(info_y))
    
    if log:
        info_X = np.log2(info_X + 1)
    
    if gene_index is None:
        gene_index = np.random.randint(info_X.shape[1])

    plt.figure(figsize=(20, 3))
    
    if group_by_classes:
        sns.boxplot(x=[data.inv_label_map[index] for index in info_y], y=info_X[:, gene_index], order=np.sort(np.unique([data.inv_label_map[index] for index in info_y])))
        plt.title(f"Expression of gene {data.genes_IDs[gene_index]} ({gene_index}) grouped by class in the dataset.")
        plt.tick_params(axis='both', which='major', labelsize=11)
        if log_scale:
            plt.yscale('log')
        plt.ylabel(f"Gene expresion\n({unit})")
    else:
        sns.boxplot(x=info_X[:, gene_index])
        plt.title(f"Expression of gene {data.genes_IDs[gene_index]} ({gene_index}) in the dataset.")
        if log_scale:
            plt.xscale('log')
        plt.xlabel(f"Gene expresion ({unit})")


def plot_random_sample_expression(data, log=False, unit='count', index=None):
    plt.figure(figsize=(20, 2))
    if index is None:
        index = np.random.randint(0, len(data))
    ID = data.sample_IDs[index]
    if log:
        plt.plot(np.log2(data.expression.loc[ID].values + 1), '.')
    else:
        plt.plot(data.expression.loc[ID].values, '.')
    plt.xlabel("Gene index")
    plt.ylabel(f"Gene expression\n({unit})")
    plt.title("Gene expression in a random sample.")      
        

def plot_stats_on_gene_expression(data, criteria='average', log_scale=False, log=False, unit='count'):
    info_X, info_y = get_X_y(data) 
    info_X = np.array(info_X)
    info_y = np.array(info_y)
    
    if log:
        info_X = np.log2(info_X + 1)
        
    if criteria == 'average':
        info_X = np.mean(info_X, axis=0)
        print(f"There are {np.sum(info_X == 0)} genes whose average expression is 0.")
    elif criteria == 'median':
        info_X = np.median(info_X, axis=0)
        print(f"There are {np.sum(info_X == 0)} genes whose median expression is 0.")
    elif criteria == 'std':
        info_X = np.std(info_X, axis=0)
        print(f"There are {np.sum(info_X == 0)} genes whose standard deviation is 0.")
    elif criteria == 'min':
        info_X = np.min(info_X, axis=0)
        print(f"There are {np.sum(info_X == 0)} genes whose minimum is 0.")
    elif criteria == 'max':
        info_X = np.max(info_X, axis=0)
        print(f"There are {np.sum(info_X == 0)} genes whose maximum is 0.")
    
    plt.figure(figsize=(20, 2))
    plt.plot(info_X, '.')
    plt.xlabel("Gene index")
    plt.ylabel(f"{criteria.capitalize()} expression\n({unit})")
    plt.show()

    plt.figure(figsize=(20, 2))
    sns.boxplot(x=info_X)
    if log_scale:
        plt.xscale('log')
    plt.xlabel(f"{criteria.capitalize()} expression per locus ({unit})")
    
    
def sort_genes(data, criteria='average', log_scale=False, unit='count'):
    info_X, info_y = get_X_y(data)
    info_X = np.array(info_X)
    info_y = np.array(info_y)
    
    if criteria == 'average':
        info_X = np.mean(info_X, axis=0)
    elif criteria == 'median':
        info_X = np.median(info_X, axis=0)
    elif criteria == 'std':
        info_X = np.std(info_X, axis=0)
    elif criteria == 'min':
        info_X = np.min(info_X, axis=0)
    elif criteria == 'max':
        info_X = np.max(info_X, axis=0)
    
    plt.figure(figsize=(20, 5))
    plt.plot(np.sort(info_X), '.')
    plt.xlabel(f"Genes sorted by {criteria.capitalize()} expression")
    plt.ylabel(f"{criteria.capitalize()} expression ({unit})")
    if log_scale:
        plt.yscale("log")
    plt.show()
    
    return np.argsort(info_X)
    

def plot_class_imbalance(data, label_name, save_path=None):
    classes = np.zeros(len(data.label_key))
    for X, y in data:
        classes[y] += 1
    xlabels = {"type": "Cancer class", "sample_type.samples": "Type"}
    plt.figure(figsize=(20, 3))
    sns.barplot(x=[label for label in data.label_key], y=[classes[data.label_map[label]] for label in data.label_key], order=np.sort([label for label in data.label_key]))
    plt.tick_params(axis='both', which='major', labelsize=11)
    plt.xlabel(f"{xlabels[label_name]}", fontsize=18)
    plt.ylabel("Number of samples", fontsize=18)
    if save_path is not None:
        plt.savefig(os.path.join(save_path, "class_imbalance"), bbox_inches='tight')
    plt.show()
    

def describe_gene_expression(data, log_scale=True, unit='count', log=False):
    info_X, info_y = get_X_y(data)
    info_X = np.array(info_X)
    info_y = np.array(info_y)
    
    if log:
        info_X = np.log2(info_X + 1)
    
    print("Mean: ", np.round(np.mean(info_X), 2))
    print("Median: ", np.round(np.median(info_X), 2))
    print("Max: ", np.round(np.max(info_X), 2))
    print("Min: ", np.round(np.min(info_X), 2))

    plt.figure(figsize=(20, 2))
    if log:
        sns.histplot(info_X.reshape(-1) + np.min(info_X), log_scale=False)
    else:
        sns.histplot(info_X.reshape(-1) + 1, binrange=[0, 7], bins=24, log_scale=True)
    plt.xlabel(f"Distribution of gene expression ({unit}) across the dataset")
    plt.ylabel("Number of\ngenes")
    if log_scale:
        plt.yscale('log')
    plt.show()
    
    plt.figure(figsize=(20, 2))
    _sum = np.sum(info_X, axis=1)
    print('Below, the gene expressions are summed per individual.')
    if log:
        sns.histplot(_sum, log_scale=False)
    else:
        sns.histplot(_sum, log_scale=False)
    plt.xlabel(f"Sum of gene expressions for each individual sample ({unit})")
    plt.ylabel("Number of\nindividuals")
    if log_scale:
        plt.yscale('log')
    plt.show()
    

def do_scatterplot_2D(X, y, labels, xlabel=None, ylabel=None, dim1=0, dim2=1, legend=True, size=2, save_name=None):
    classes = np.unique(y)
    cmap = sns.color_palette(palette=cc.glasbey, n_colors=len(classes))
    
    plt.figure(figsize=(4, 4))
    for i, _class in enumerate(classes):
        plt.scatter(X[y[:, 0]==_class, dim1], X[y[:, 0]==_class, dim2], color=cmap[i], label=labels[_class], s=size)
    if legend:
        plt.legend(ncol=1, bbox_to_anchor=(1, 0.9), markerscale=4.)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if save_name:
        plt.savefig(save_name, bbox_inches='tight', dpi=150)
    plt.show()
    
    
    
def plot_box(data, xlabel=None, save_name=None):
    plt.figure(figsize=(15, 4))
    plt.boxplot(data, vert=False)
    plt.yticks(color='w')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    if xlabel:
        plt.xlabel(xlabel)
    if save_name:
        plt.savefig(save_name, bbox_inches='tight', dpi=150)
    plt.show()
    
    
def describe_random_individuals(data, log=True, log_scale=False, save_path=None, unit='log2(count+1)'):
    info_X, info_y = get_X_y(data)
    info_X = np.array(info_X)
    info_y = np.array(info_y)

    plt.figure(figsize=(15, 4))

    my_samples = {}
    for i in range(20):
        j = np.random.randint(info_X.shape[0])
        if log:
            my_samples[i] = np.log2(info_X[j]+1)
        else:
            my_samples[i] = info_X[j]

    plt.boxplot(my_samples.values(), vert=True)
    if log_scale:
        plt.yscale('log')
    plt.xticks(color='w')
    plt.xlabel('Distribution of gene expression in 20 individuals (boxes)')
    plt.ylabel(f'Gene expression\n({unit})')
    if save_path is not None:
        plt.savefig(os.path.join(save_path, "random_individuals"), bbox_inches='tight')
    plt.show()
    

def plot_draw_from_Dirichlet(alpha, sparse=False, label_distrib='Class', label_var='Pathway'):
    '''
    Draw an observation from a Dirichlet distribution. 
    Parameter:
       alpha  --  dict, for each key, contains a vector used to parameterized a Dirichlet distribution.
    '''
    # Parameters
    n_draw = len(alpha)
    n_var = len(alpha[list(alpha.keys())[0]])
    # Definition of the colors used for the plot
    if n_var < 4:
        colors = ['violet', 'royalblue', 'springgreen']
    else:
        cmap = cm.get_cmap('viridis', n_var)
        colors = np.arange(n_var)
        np.random.seed(0)
        np.random.shuffle(colors)
        np.random.seed()
        colors = [cmap(c) for c in list(colors)]
    # For each distribution...
    fig, ax = plt.subplots()
    for c, draw in enumerate(alpha.keys()):
        # ...Get data
        if not sparse:
            data = np.random.dirichlet(alpha[draw])
            print(f"{label_distrib} {c}: {alpha[draw] / np.sum(alpha[draw])}")
        else:
            data = alpha[draw].copy()
            values = np.random.dirichlet(np.ones(np.sum([alpha[draw]!=0])))
            data[data!=0] = values
            print(f"{label_distrib} {c}: {data}")
        # ...Plot
        if c == 0:
            ax.barh(c, data[0], label=f'{label_var} 0', color=colors[0])
        else:
            ax.barh(c, data[0], color=colors[0])
        bottom = data[0]
        for p in range(1, n_var):
            if c == 0:
                ax.barh(f'{label_distrib} {c}', data[p], left=bottom, label=f'{label_var} {p}', color=colors[p])
            else:
                ax.barh(f'{label_distrib} {c}', data[p], left=bottom, color=colors[p])
            bottom += data[p]
    ax.set_ylabel('Proportions')
    ax.set_xticklabels([])
    ax.invert_yaxis()
    if n_var < 5:
        ax.legend()
    plt.show()


