import numpy as np
import torch
import os
from evaluate import *
from sklearn.metrics import classification_report, balanced_accuracy_score



# Useful functions
def sort_attribution_scores(attr):
    return np.argsort(-np.abs(attr))

    
def get_attributions_per_class(attr, labels, _class, method=None, value=None):
    """
    Return the normalized attributions associated with the examples associated with the label _class.
    """
    indices = np.argwhere(labels == _class)[:, 0]
    attr_cls = attr[indices]
    return normalize_attribution_scores(attr_cls, method, value)


def normalize_attribution_scores(attr, method, value=None):
    """
    To globally interpret local explanations, the scores are normalized per feature over all inputs.
    """
    assert method in ['mean', 'quantile'], "`method` must be 'mean' or 'quantile'"
    if method == 'quantile':
        assert value is not None and value >=0 and value <= 1, "when method=='quantile', the value must be a floating value between 0 and 1."
        attr = torch.quantile(attr, value, dim=0)
    elif method == 'mean':
        attr = attr.mean(0)
    return attr


def get_number_common_elements(list1, list2):
    return len(list(set(list1).intersection(list2)))


def get_common_genes(list1, list2, interval):
    nb = []
    for gap in interval:
        nb.append(get_number_common_elements(list1[:gap], list2[:gap]))
    return nb


def sort_features(reference, order, scores):
    """
    Return the indices of the features ordered by importance for a given reference.
    
    Parameters:
        reference  --  "general" or a number between 0 and 32 representing a class
        order  -- "increasing", features ordered with increasing importance (default: decreasing)
    """
    sorted_indices = scores[reference]['sorted_indices']  
    if order == 'increasing':
        sorted_indices = sorted_indices[::-1].copy()
    return sorted_indices


def remove_features(X, sorted_features, number):
    """Set to 0 the first `number` sorted_features in X.
    """
    zeros = sorted_features[:number]
    X_temp = X.clone()
    device = X_temp.device
    X_temp[:, zeros] = torch.zeros((X.shape[0], number)).to(device)
    return X_temp


def keep_features(X, best_indices):
    """Set to 0 the features which are not in best_indices.
    """
    zeros = np.array(list(set(np.arange(0, X.shape[1], 1)) - set(best_indices)))
    X_temp = X.clone()
    device = X_temp.device
    X_temp[:, zeros] = torch.zeros((X.shape[0], len(zeros))).to(device)
    return X_temp


def get_metrics(model, X, y, labels_name=None, get_cm=True):
    """Given a model, input data X and labels y, return the accuracy, the classification report and a confusion matrix.    
    """
    outputs = model(X)
    _, y_pred = torch.max(outputs.data, 1)
    y_pred = y_pred.cpu().detach().numpy()
    y = y.cpu().detach().numpy()
    acc = compute_accuracy_from_predictions(y_pred, y)
    b_acc = np.round(balanced_accuracy_score(y, y_pred)*100, 2)
    _dict = classification_report(y, y_pred, output_dict=True, zero_division=0)
    if get_cm:
        cm = get_confusion_matrix(y, y_pred, labels_name, normalize='true')
        return b_acc, acc, _dict, cm
    else:
        return b_acc, acc, _dict


def get_metrics_with_removed_features(model, X, y, classes, sorted_features, nb_to_remove, feat_to_remove=[]):
    """
    After removing some features, the accuracy, precision, recall are computed. 
    
    Parameters:
        model  --  neural network
        X  --  input, torch tensor (batch_size, n_feat)
        y  --  labels, torch tensor (batch_size)
        classes  -- list containing the classes to consider
        sorted_features  -- indices of the features, the first ones are removed first.
        nb_to_remove -- list containing the numbers of features to remove from X
        feat_to_remove  --  [], list containing lists of features to remove from X (the additional ones are removed randomly)
    """
    balanced_accuracy = []
    accuracy = []
    recall = {}
    precision = {}
    
    for _class in classes:
        recall[_class] = []
        precision[_class] = []
        
    for i, nb in enumerate(nb_to_remove):
        if feat_to_remove != []:
            features = list(feat_to_remove[i]) + [feat for feat in sorted_features if feat not in feat_to_remove[i]]
        else:
            features = sorted_features.copy()
        X_temp = remove_features(X, features, nb)
        b_acc, acc, _dict = get_metrics(model, X_temp, y, None, get_cm=False)
        balanced_accuracy.append(b_acc)
        accuracy.append(acc)
        
        for _class in classes:
            recall[_class].append(_dict[str(_class)]['recall'])
            precision[_class].append(_dict[str(_class)]['precision'])
        
    return balanced_accuracy, accuracy, recall, precision



def get_metrics_with_selected_features(model, X, y, feat_to_keep, classes=[]):
    """
    Return the performance of the model after setting some features to 0. 
    
    Parameters:
        model  --  neural network
        X  --  input, torch tensor (batch_size, n_feat)
        y  --  labels, torch tensor (batch_size)
        feat_to_keep  -- indices of the features to keep (the others are set to 0)
    """
    recall = {}
    precision = {}

    X_temp = keep_features(X, feat_to_keep)
    balanced_accuracy, accuracy, _dict = get_metrics(model, X_temp, y, None, get_cm=False)

    for _class in classes:
        recall[_class] = _dict[str(_class)]['recall']
        precision[_class] = _dict[str(_class)]['precision']
        
    return balanced_accuracy, accuracy, recall, precision



# Global representation (per class)
def get_results_per_class(model, X, y, K, scores, setting='keep_best', classes=[]):
    """
    Compute several metrics measuring the performance of 'model' when the top-k features per class are kept (others features set to 0) or removed (set to 0).
    Return 
        n_kept  --  a list with the number of features kept
        kept_feat  --  a list with the indices of features kept
        results  --  a dictionary containing
            results['balanced_accuracy']  --  a list of balanced accuracy
            results['accuracy'] --  a list of accuracy
            results[_class]['recall']  --  a list of recall for a _class specified in `classes`
            results[_class]['precision']  --  a list of precision for a _class specified in `classes`
    
    Parameters:
        model  --  neural network
        X  --  input, torch tensor (batch_size, n_feat)
        y  --  labels, torch tensor (batch_size)
        K -- list of integers k, numbers of best features per class to keep or remove
        scores  --  dict, scores[_class] contains two lists scores[_class]['attr'], scores[_class]['sorted_indices'].
                    scores[_class]['attr'] is a list of floats indicating the importance of a feature.
                    scores[_class]['sorted_indices'] is a list of indices sorting the 'attr' with decreasing absolute values.
        setting  --  'keep_best' or 'remove_best', indicate whether the best features are kept or removed
        classes  -- None or list containing the classes to consider (for precision and recall)
    """
    assert setting in ['keep_best', 'remove_best']
    
    n_kept = []
    kept_feat = []
    results = {}
    results['accuracy'] = []
    results['balanced_accuracy'] = []
    for _class in classes:
        results[_class] = {}
        results[_class]['recall'] = []
        results[_class]['precision'] = []
        
    for k in K:
        
        if setting == 'keep_best':
            feat_to_keep = keep_best_features(k, scores)
        else:
            feat_to_keep = list(set(np.arange(X.shape[1])) - set(keep_best_features(k, scores)))
             
        n_kept.append(len(feat_to_keep))
        kept_feat.append(feat_to_keep)

        balanced_accuracy, accuracy, recall, precision = get_metrics_with_selected_features(model, X, y, feat_to_keep)

        results['accuracy'].append(accuracy)
        results['balanced_accuracy'].append(balanced_accuracy)

        for _class in classes:
            results[_class]['recall'].append(recall[_class])
            results[_class]['precision'].append(precision[_class])

    return results, n_kept, kept_feat


def keep_best_features(nb_per_class, scores):
    """
    Return the indices of the nb_per_class most important features.
    
    Parameters:
        nb_per_class -- number of features to keep per class
        scores  --  dict, scores[_class] contains two lists scores[_class]['attr'], scores[_class]['sorted_indices'].
                    scores[_class]['attr'] is a list of floats indicating the importance of a feature.
                    scores[_class]['sorted_indices'] is a list of indices sorting the 'attr' with decreasing absolute values.
    """
    n_class = len(scores.keys()) - 1
    indices = []
    for _class in range(n_class):
        indices.extend(scores[_class]['sorted_indices'][:nb_per_class])
    return np.unique(indices)



# Local representation (per sample)
def get_results_with_best_features_kept_or_removed(model, X, y, K, attr, classes=[], kept=True, balance=False):
    """
    Compute several metrics measuring the performance of 'model' when the top-k best features are removed (set to 0) or 
    kept (all the others set to 0).
    
    Return 
        kept_feat  --  a list with the indices of features kept
        results  --  a dictionary containing
            results['balanced_accuracy']  --  a list of balanced accuracy
            results['accuracy'] --  a list of accuracy
            results[_class]['recall']  --  a list of recall for a _class specified in `classes`
            results[_class]['precision']  --  a list of precision for a _class specified in `classes`
            
    Parameters:
        model  --  neural network
        X  --  input, torch tensor (batch_size, n_feat)
        y  --  labels, torch tensor (batch_size)
        K -- list of integers k, numbers of best features to keep or remove
        attr  --  importance of the features per sample, array (batch_size, n_feat)
        classes  -- None or list containing the classes to consider (for precision and recall)
        kept  --  True or False, indicate whether the features are kept or removed
        balance  -- True or False, indicate whether the average importance of a feature is balanced wrt the class imbalance
    """
    kept_feat = []
    results = {}
    results['accuracy'] = []
    results['balanced_accuracy'] = []
    for _class in classes:
        results[_class] = {}
        results[_class]['recall'] = []
        results[_class]['precision'] = []
    
    if balance:
        n_class = len(torch.unique(y))
        n_feat = X.shape[1]
        scores = np.zeros(n_feat)
        for _class in range(n_class):
            scores += np.mean(np.abs(attr[y.cpu() == _class]), axis=0)
        scores = scores / n_class
    else:
        scores = np.mean(np.abs(attr), axis=0)
    
    if kept:
        indices = np.argsort(-scores)
    else:
        indices = np.argsort(scores)
    
    for k in K:
        feat_to_keep = indices[:k]
        kept_feat.append(feat_to_keep)
        balanced_accuracy, accuracy, recall, precision = get_metrics_with_selected_features(model, X, y, feat_to_keep)

        results['accuracy'].append(accuracy)
        results['balanced_accuracy'].append(balanced_accuracy)
            
        for _class in classes:
            results[_class]['recall'].append(recall[_class])
            results[_class]['precision'].append(precision[_class])

    return results, kept_feat



# Random
def get_results_with_random_features(model, X, y, nb_to_keep, n_simu, classes=[], feat_to_remove=[]):
    """
    Return the accuracy and balanced accuracy of 'model' when random features are set to 0. 
    Also return the precision and recall for the specified 'classes'.
    
    Parameters:
        model  --  neural network
        X  --  input, torch tensor (batch_size, n_feat)
        y  --  labels, torch tensor (batch_size)
        nb_to_keep -- list containing the numbers of features to keep from X
        n_simu  --  int, number of simulations the results are computed on
        classes  -- None or list containing the classes to consider (for precision and recall)
        feat_to_remove  --  None, list containing lists of features to remove from X (the additional ones are removed randomly)
    """
    # Initialize the outputs
    results = {}
    results['balanced_accuracy'] = []
    results['accuracy'] = []
    for _class in classes:
        results[_class] = {}
        results[_class]['recall'] = []
        results[_class]['precision'] = []
    
    # Compute the outputs for n_simu
    n_feat = X.shape[1]
    random_features = np.arange(0, n_feat, 1)
    for i in range(n_simu):
        np.random.shuffle(random_features)
        balanced_accuracy, accuracy, recall, precision = get_metrics_with_removed_features(model, X, y, classes, random_features, n_feat - nb_to_keep, feat_to_remove)
        results['balanced_accuracy'].append(balanced_accuracy)
        results['accuracy'].append(accuracy)
        for _class in classes:
            results[_class]['recall'].append(recall[_class])
            results[_class]['precision'].append(precision[_class])
    
    # Average the outputs
    final_results = {}
    for term in ['balanced_accuracy', 'accuracy']:
        final_results[term] = {}
        final_results[term]['mean'] = np.mean(results[term], axis=0)
        final_results[term]['std'] = np.std(results[term], axis=0)
    for _class in classes:
        final_results[_class] = {}
        for term in ['recall', 'precision']:
            final_results[_class][term] = {}
            final_results[_class][term]['mean'] = np.mean(results[_class][term], axis=0)
            final_results[_class][term]['std'] = np.std(results[_class][term], axis=0)
    return final_results



# Prediction gaps
def prediction_gap_with_dataloader(model, loader, transform, attr, n_class, _type, y_true, y_pred, int_gap):
    # Informations
    n_sample = len(loader.sampler)
    x, _ =  next(iter(loader))
    n_feat = x.shape[1]
    device = model.fc.weight.device

    # Compute the gaps
    PG = np.zeros(n_class)
    count_per_class = np.zeros(n_class)

    torch.manual_seed(1)
    count = 0
    for i, (x, y) in enumerate(loader):
        print(i, end='\r')
        batch_size = x.shape[0]

        x = x.to(device)
        y = y.numpy()
        if transform:
            x = transform(x)

        # Check data order with true classes
        assert (y == y_true[count:count + batch_size]).all(), 'Problem with data order.'

        # Check model 
        _class = torch.argmax(model(x), axis=1).cpu().numpy() * 1.
        assert (_class == y_pred[count:count + batch_size]).all(), 'Problem with model.'

        # Rank features by importance
        if _type == 'important':
            indices = np.argsort(-attr[count:count + batch_size], axis=1)
        elif _type == 'unimportant':
            indices = np.argsort(attr[count:count + batch_size], axis=1)

        # Prediction gap
        n_point = int(n_feat / int_gap)
        pred_gap = np.zeros((batch_size, n_point))
        pred_full = model(x)[np.arange(batch_size), y].detach().cpu().numpy()

        for i in range(n_point):
            s = np.repeat(np.arange(batch_size), int_gap)
            f = indices[:, i*int_gap:i*int_gap+int_gap].reshape(-1)
            x[s, f] = torch.zeros((batch_size * int_gap)).to(device)
            pred = model(x)
            pred = pred[np.arange(batch_size), y].detach().cpu().numpy()
            pred_gap[:, i] = pred_full - pred
        mask = (pred_gap) > 0 * 1.0
        copy = pred_gap.copy()
        pred_gap = np.sum(mask * pred_gap, axis=1)/n_feat*int_gap

        # Sum without taking into account misclassified examples
        correct_indices = (_class == y)
        for s in range(batch_size):
            if _class[s] == y[s]:
                PG[y[s]] += np.sum(pred_gap[s])
                count_per_class[y[s]] += 1

        # Update count
        count += batch_size

    # Average PG per class
    PG = PG / count_per_class

    return PG

