# XAI_for_genomics

This repository contains the code of the experiments performed in the following paper\
[Studying Limits of Explainability by Integrated Gradients for Gene Expression Models](https://arxiv.org/pdf/2303.11336v1.pdf)
by Myriam Bontonou, Anaïs Haget, Maria Boulougouri, Jean-Michel Arbona, Benjamin Audit, Pierre Borgnat.

This repository also contains the code of the additional results presented in the French article\
[Expliquer la classification d'expression de gènes par la méthode des gradients intégrés]().


## Abstract
Understanding the molecular processes that drive cellular life is a fundamental question in biological research. Ambitious programs have gathered a number of molecular datasets on large populations. To decipher the complex cellular interactions, recent work has turned to supervised machine learning methods. The scientific questions are formulated as classical learning problems on tabular data or on graphs, e.g. phenotype prediction from gene expression data. In these works, the input features on which the individual predictions are predominantly based are often interpreted as indicative of the cause of the phenotype, such as cancer identification.
Here, we propose to explore the relevance of the biomarkers identified by Integrated Gradients, an explainability method for feature attribution in machine learning. Through a motivating example on The Cancer Genome Atlas, we show that ranking features by importance is not enough to robustly identify biomarkers. As it is difficult to evaluate whether biomarkers reflect relevant causes without known ground truth, we simulate gene expression data by proposing a hierarchical model based on Latent Dirichlet Allocation models. We also highlight good practices for evaluating explanations for genomics data and propose a direction to derive more insights from these explanations.

## Usage
### 1. Dependencies
- Python = 3.7
- PyTorch = 1.11
- PyTorch geometric = 2.0


### 2. Datasets
The datasets will be stored in a folder on your computer. Set the absolute path of this folder in the function set_path in setting.py.

#### TCGA data
##### PanCan
To download the PanCan TCGA dataset [1], go to the Pancan/Data folder and execute `python get_pancan.py`.
More details on the data are presented in two notebooks `Describe_tcga_data.ipynb` and `Discover_gene_expression_data.ipynb`.

##### BRCA and KIRC
To download the BRCA and KIRC datasets, go to the Gdc/Data folder and execute `python get_gdc.py`.
More details on the data are presented in two notebooks `Describe_gdc_tcga_data.ipynb` and `Discover_gdc_tcga_gene_expression_data.ipynb`.

#### Simulation
To generate SIMU1/SIMU2 data, go to Simulation/Data folder and execute `python get_simu.py --name SIMU1 --size 9900` and `python get_simu.py --name SIMU2 --size 9900`.

To generate SimuA/SimuB/SimuC data, execute `python get_simu.py --name SimuA --size 1200`, `python get_simu.py --name SimuB --size 1200` and `python get_simu.py --name SimuC --size 1200`.


### 3. Learning models
In the following, the same commands can be used for various datasets and learning models.
- Various datasets can be used: PanCan TCGA (pancan), BRCA, KIRC, SIMU1/SIMU2, SimuA/SimuB/SimuC. 
- Various models can be trained: logistic regression (LR), multilayer perceptron (MLP), diffusion + logistic regression (DiffuseLR), diffusion + multilayer perceptron (DiffuseMLP).

Go to Scripts/Model.
#### Graph
To compute the correlation graph over all features, execute `python infer_graph.py --name pancan`.

#### Model
To train a logistic regression (LR) on TCGA data (pancan), execute `python train_nn.py -n pancan -m LR`.


### 4. Explainability
Go to Scripts/Explanation.

To compute the integrated gradients scores, execute `python get_attributions.py -n pancan -m LR --set train` for the training examples and `python get_attributions.py -n pancan -m LR --set test` for the test examples.

To compute the prediction gaps (PGs), execute `python get_prediction_gaps.py -n pancan -m LR --set test`. Local PGs are obtained by ranking the features of each example independently. Global PGs are obtained by ranking them in the same way for all examples of the same class.

To compute the curves, execute `python get_attributions_averaged_per_class.py -n pancan -m LR --set test` followed by `python get_curves.py -n pancan -m LR --set test --simu 100`.

To compute the feature agreement metrics on simulated data, execute `get_ranking_metrics.py -n SIMU1 -m LR`. To compute the diffused feature agreement metrics on simulated data, execute `get_ranking_metrics.py -n SIMU1 -m LR --diffusion`.

## References
[1] The data come from the [TCGA Research Network](https://www.cancer.gov/tcga).

## Contact
Please contact us if there are any problems.

Myriam Bontonou (myriam.bontonou@ens-lyon.fr)
