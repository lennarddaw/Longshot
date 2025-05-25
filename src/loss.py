import numpy as np

def mse_loss(y_pred, y_true):
    m = y_true.shape[1]
    loss = np.sum((y_pred - y_true)**2) / (2*m)
    return loss

def mse_derivative(y_pred, y_true):
    m = y_true.shape[1]
    return (y_pred - y_true) / m
