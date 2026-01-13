import numpy as np
import torch
from sklearn.metrics import roc_auc_score

RANDOM_SEED = 123

def sigmoid(z):
    return 1. / (1. + np.exp(-z))


def int_to_onehot(y, num_labels):
    ary = np.zeros((y.shape[0], num_labels))
    for i, val in enumerate(y):
        ary[i, val] = 1
    return ary


def minibatch_generator(X, y, minibatch_size):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    for start_idx in range(0, indices.shape[0] - minibatch_size 
                           + 1, minibatch_size):
        batch_idx = indices[start_idx:start_idx + minibatch_size]
        
        yield X[batch_idx], y[batch_idx]


def calculate_macro_auc(model, X, y, num_classes, model_type='numpy'):
    if model_type == 'numpy_one':
        _, a_out = model.forward(X)
        probabilities = a_out
    elif model_type == 'numpy_two':
        _, _, a_out = model.forward(X)
        probabilities = a_out
    elif model_type == 'pytorch':
        model.eval()
        with torch.no_grad():
            x_torch = torch.DoubleTensor(X)
            outputs = model(x_torch)
            probabilities = outputs
    
    y_onehot = int_to_onehot(y, num_classes)
    macro_auc = roc_auc_score(y_onehot, probabilities, average='macro', multi_class='ovr')
    
    return macro_auc
