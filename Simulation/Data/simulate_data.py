import numpy as np


def generate_hierarchical_data(alpha, eta, n_sample, proportion=None):
    # Parameters
    n_class = len(alpha.keys())
    n_pathway = len(eta.keys())
    n_gene = len(eta[list(eta.keys())[0]])
    n_read = 1500000
    
    # Generate data
    X = np.zeros((n_sample, n_gene))
    y = np.zeros((n_sample), dtype='int')

    count = 0
    for c in range(n_class):
        # Number of samples per class
        if proportion is None:
            n_sample_per_class = int(n_sample/n_class)
        else:
            n_sample_per_class = int(proportion[c] * n_sample)
        genes = np.zeros((n_sample_per_class, n_gene))
        # Labels (n_sample)
        y[count:count+n_sample_per_class] = np.ones((n_sample_per_class), dtype='int') * c
        # Proportion of pathways in a sample (n_sample, n_pathway)
        class_to_path = np.random.dirichlet(alpha['C'+str(c)], size=(n_sample_per_class))
        # Each sample contains n_read reads associated with genes.
        for sample in range(n_sample_per_class):
            print(count+sample, end='\r')
            # For each read, draw a pathway
            pathway = np.random.multinomial(n_read, class_to_path[sample])
            # For each read, draw a gene
            for p in range(n_pathway):
                reads = np.random.multinomial(pathway[p], eta['P'+str(p)])
                genes[sample] += reads
        genes /= (n_read/1000000)    
        X[count:count+n_sample_per_class, :] = genes
        count += n_sample_per_class
    return X, y


def generate_eta(n_pathway, sparsity, n_gene, case=None):
    eta = {}
    for p in range(n_pathway):
        # Define the underlying graph structure
        if case == 1:
            eta['P'+str(p)] = np.array([0.,] * 10 * p + [1.,] * 10 + [0.,] * 10 * (n_pathway-p-1)) * 5
        elif case == 0:
            eta['P'+str(p)] = np.random.binomial(1, sparsity, size=[n_gene]) * 5.
            while np.sum(eta['P'+str(p)]) < 3:
                eta['P'+str(p)] = np.random.binomial(1, sparsity, size=[n_gene]) * 5.
        # Attribute weights to each edge
        values = np.random.dirichlet(eta['P'+str(p)][eta['P'+str(p)]!=0], size=(1))
        eta['P'+str(p)][eta['P'+str(p)]!=0] = values.reshape(-1)
    return eta


def return_parameters(name):
    n_class = 33
    n_gene = 15000
    proportion = None  # if None, generate a balanced number of samples per class
    alpha = {}
    
    # General
    if name == 'SIMU1':
        n_pathway = 1500
        for c in range(n_class):
            alpha['C' + str(c)] = np.array([1.] * n_pathway)   # each pathway has a priori the same importance
    elif name == 'SIMU2':
        n_pathway = 3000
    for c in range(n_class):
        alpha['C' + str(c)] = np.array([1.] * n_pathway)

    # Number of overexpressed pathways
    useful_paths = []
    cls_gap = 2.
    P = 37
    useful_paths = {}
    for c in range(n_class):
        useful_paths['C' + str(c)] = []                
        for p in range(P):
            alpha['C' + str(c)][P * c + p] = cls_gap
            useful_paths['C' + str(c)].append('P' + str(P * c + p))
            
    # Variance
    for c in range(n_class):
        alpha['C'+str(c)] = alpha['C'+str(c)] * 4.
        
    # Prior on gene distribution per pathway   
    if name == 'SIMU1':
        case = 1
        sparsity = None
    elif name == 'SIMU2':
        case = 0
        sparsity = 1 / n_gene * 10
    

    # Drawn gene distribution per pathway
    eta = generate_eta(n_pathway, sparsity, n_gene, case=case)

    # Store important genes
    useful_genes = {}
    for c in range(n_class):
        for P in (useful_paths["C" + str(c)]):
            useful_genes[P] = np.argwhere(eta[P] != 0).reshape(-1)

    # Check validity (useful genes must have a drawing probability higher than 0.01)
    for P in useful_genes.keys():
        print('Pathway', P, end='\r')
        validity = False
        while not validity:
            if min(eta[P][useful_genes[P]]) >= 0.01:
                validity = True
            else:
                eta[P] = generate_eta(n_pathway, sparsity, n_gene, case=case)[P]
                useful_genes[P] = np.argwhere(eta[P] != 0).reshape(-1)
            
    return alpha, eta, proportion, n_gene, n_pathway, n_class, useful_paths, useful_genes

