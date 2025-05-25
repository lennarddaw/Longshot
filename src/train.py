import numpy as np
from utils import create_batches, train_test_split
from activations import sigmoid, sigmoid_derivative
from layers import Dense
from loss import mse_loss, mse_derivative
from optimizers import SGD
from network import NeuralNetwork

def load_your_data():
    # TODO: Replace with actual data loading logic
    # Example: Generate dummy data for demonstration
    X = np.random.rand(10, 100)  # 10 features, 100 samples
    y = np.random.rand(1, 100)   # 1 output, 100 samples
    return X, y

# 1. Daten vorbereiten
X, y = load_your_data()
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

# 2. Modell bauen
nn = NeuralNetwork()
nn.add(Dense(in_dim=X.shape[0], out_dim=64, activation=sigmoid, activation_derivative=sigmoid_derivative))
nn.add(Dense(in_dim=64, out_dim=y.shape[0], activation=sigmoid, activation_derivative=sigmoid_derivative))

# 3. Training
optimizer = SGD(lr=0.1)
epochs = 1000
batch_size = 32

for epoch in range(epochs):
    batches = create_batches(X_train, y_train, batch_size)
    for X_batch, y_batch in batches:
        nn.X_input = X_batch; nn.m = X_batch.shape[1]
        y_pred = nn.forward(X_batch)
        grads = nn.backward(y_pred, y_batch)
        nn.update_params(grads, optimizer)

    # Optionale Validierung
    if epoch % 100 == 0:
        y_val_pred = nn.forward(X_val)
        val_loss = mse_loss(y_val_pred, y_val)
        print(f"Epoch {epoch}: Validierungs-Loss = {val_loss:.4f}")
