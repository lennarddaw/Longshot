from src.loss import mse_derivative


class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, X):
        A = X
        for layer in self.layers:
            A = layer.forward(A)
        return A

    def backward(self, y_pred, y_true):
        dA = mse_derivative(y_pred, y_true)
        grads = []
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            A_prev = self.layers[i-1].A if i>0 else self.X_input
            dA, dW, db = layer.backward(dA, A_prev, self.m)
            grads.insert(0, (dW, db))
        return grads

    def update_params(self, grads, optimizer):
        for layer, (dW, db) in zip(self.layers, grads):
            layer.W = optimizer.update(layer.W, dW)
            layer.b = optimizer.update(layer.b, db)
