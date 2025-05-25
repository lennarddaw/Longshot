import numpy as np
from torch import relu

class Dense:
    def __init__(self, in_dim, out_dim, activation, activation_derivative):
        self.W = np.random.randn(out_dim, in_dim) * 0.01
        self.b = np.zeros((out_dim, 1))
        self.activation = activation
        self.activation_derivative = activation_derivative

    def forward(self, A_prev):
        self.Z = self.W @ A_prev + self.b
        self.A = self.activation(self.Z)
        return self.A

    def backward(self, dA, A_prev, m):
        dZ = dA * self.activation_derivative(self.A if self.activation!=relu else self.Z)
        dW = (1/m) * dZ @ A_prev.T
        db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = self.W.T @ dZ
        return dA_prev, dW, db
