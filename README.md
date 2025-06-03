# Longshot

Longshot is a lightweight Python library for building and training simple feed‐forward neural networks **from scratch**—without relying on heavy deep‐learning frameworks. It exposes every component (activations, layers, losses, optimizers, training loop) in pure Python/NumPy, making it ideal for educational purposes, debugging, and rapid prototyping of novel network components and optimization strategies.

---

## 🚀 Features

- **Pure Python/NumPy Implementation**  
  All core building blocks (activations, layers, loss functions, optimizers) are implemented in Python and depend only on NumPy. No external DL libraries or “magic” abstractions.

- **Modular Layer & Activation API**  
  - `Dense` (fully connected) layers  
  - Built‐in activation functions (ReLU, Sigmoid, Tanh, etc.)  
  - Easy to extend: write your own `forward`/`backward` logic and plug it into the network.

- **Customizable Loss Functions**  
  - Mean Squared Error (MSE)  
  - Cross‐Entropy  
  - Add new loss / gradient rules as needed.

- **Optimizers**  
  - Stochastic Gradient Descent (SGD)  
  - Adam  
  - Extendable: implement AdamW, RMSprop, or any optimization rule.

- **Training Loop & Utilities**  
  - Mini‐batch generator, shuffling, and epoch management  
  - Automatic forward/backward pass coordination  
  - Data‐handling helpers (weight initialization, accuracy metrics, etc.)

- **Educational Transparency**  
  Every line of code is visible and traceable. Ideal for understanding how backpropagation, gradient computation, and weight updates are handled “under the hood.”

---

## 📁 Repository Structure

Longshot/
└── src/
├── init.py
├── activations.py # Activation functions + their derivatives
├── layers.py # Layer definitions (Dense, Dropout, etc.)
├── loss.py # Loss functions (MSE, Cross-Entropy, etc.)
├── network.py # Network class: orchestrates forward/backward passes
├── optimizers.py # Optimizer implementations (SGD, Adam, …)
├── train.py # Training loop: batching, forward, backward, updates
└── utils.py # Utilities (data loaders, initialization, metrics)

markdown
Kopieren
Bearbeiten

- `activations.py`  
  Defines classes/functions for ReLU, Sigmoid, Tanh, Softmax, etc., along with their derivatives (for backprop).

- `layers.py`  
  Implements `Dense` (fully connected) and other layers (e.g., Dropout). Each layer has `forward()` and `backward()` methods to compute activations and gradients, respectively.

- `loss.py`  
  Contains loss/criterion classes like `MSELoss` and `CrossEntropyLoss`, each with a `forward(predictions, targets)` and `backward()` to compute loss gradients.

- `network.py`  
  Provides a `Network` class that manages a sequence of layers. It handles:
  1. Forward propagation through all layers  
  2. Accumulating gradients during backpropagation  
  3. Passing gradients to the optimizer for weight updates

- `optimizers.py`  
  Implements optimization algorithms:
  - **SGD**: vanilla stochastic gradient descent  
  - **Adam**: Adaptive Moment Estimation  
  Each optimizer takes `model.parameters()` as input and updates them according to its rule.

- `train.py`  
  Houses a `Trainer` class (or function) that:
  1. Splits data into mini‐batches  
  2. Executes forward pass to compute predictions  
  3. Computes loss and calls `.backward()` on loss to populate gradients  
  4. Calls optimizer to update layer weights  
  5. Tracks epoch‐level metrics (loss, accuracy)

- `utils.py`  
  Miscellaneous helper functions:
  - Weight‐initialization routines (e.g. random normal, Xavier)  
  - Data‐shuffling and batching utilities  
  - Accuracy / metric calculators  
  - Any other small routines that simplify experiments

---

## 🚀 Getting Started

### 1. Prerequisites

- Python 3.7+  
- [NumPy](https://numpy.org/)  

> **Note:** Longshot is intentionally minimal. It relies only on NumPy for numerical operations. You may install additional libraries (e.g., Matplotlib, pandas) for data preprocessing or visualization, but they are not required by Longshot itself.

### 2. Installation

1. **Clone the repository**  
   ```bash
   git clone https://github.com/lennarddaw/Longshot.git
   cd Longshot/src
Install dependencies

bash
Kopieren
Bearbeiten
pip install numpy
3. Quick Example: Train a Small Network
Below is a simple script that trains a one‐hidden‐layer neural network to approximate a linear function (y = 2x + noise).

python
Kopieren
Bearbeiten
import numpy as np

from activations import ReLU, Sigmoid
from layers import Dense
from loss import MSELoss
from optimizers import SGD
from network import Network
from train import Trainer

# 1. Create synthetic data
np.random.seed(42)
X = np.random.randn(1000, 1)           # 1D input
y = 2 * X + 0.5 * np.random.randn(1000, 1)

# 2. Define a simple feed‐forward network
model = Network([
    Dense(input_dim=1, output_dim=16, activation=ReLU()),
    Dense(input_dim=16, output_dim=1, activation=Sigmoid())
])

# 3. Choose loss and optimizer
criterion = MSELoss()
optimizer = SGD(model.parameters(), lr=0.01)

# 4. Set up Trainer and train for 50 epochs
trainer = Trainer(
    model=model,
    loss_fn=criterion,
    optimizer=optimizer,
    epochs=50,
    batch_size=32,
    verbose=True
)

trainer.train(X, y)

# 5. After training, make a prediction
test_input = np.array([[0.7]])
prediction = model.forward(test_input)
print(f"Prediction for x=0.7: {prediction.flatten()[0]:.4f}")
Explanation of Steps
Data Generation

We sample 1,000 points X ~ N(0, 1) and build targets y = 2x + noise.

Model Definition

Network([ … ]) takes a Python list of layer instances.

Each Dense layer requires input_dim, output_dim, and an activation instance (e.g. ReLU()).

Loss & Optimizer

MSELoss() calculates mean squared error and its gradient.

SGD(model.parameters(), lr=0.01) updates all parameters in the network with a fixed learning rate.

Training Loop

Trainer orchestrates batching, forward pass, backward pass, weight updates, and logging.

trainer.train(X, y) runs multiple epochs, printing training loss/accuracy per epoch if verbose=True.

Inference

After training, call model.forward(...) on new inputs to get predictions.

🔍 How to Extend
Longshot is built for extensibility. Here are common extension points:

1. Add a New Activation Function
Open activations.py.

Create a new class, e.g., class LeakyReLU(Activation): implementing:

forward(self, x) → out

backward(self, grad_output) → grad_input

Use it in a layer:

python
Kopieren
Bearbeiten
from activations import LeakyReLU
my_layer = Dense(input_dim=32, output_dim=32, activation=LeakyReLU(alpha=0.01))
2. Create a Custom Layer
Open layers.py.

Subclass a base Layer (if defined) or follow the structure of Dense:

python
Kopieren
Bearbeiten
class CustomLayer:
    def __init__(self, ...):
        # Initialize weights, biases, etc.
    def forward(self, input_data):
        # Compute output and store anything needed for backward
        return output
    def backward(self, grad_output):
        # Compute gradients w.r.t. inputs and parameters
        # Store parameter gradients in self.grads
        return grad_input
    def parameters(self):
        # Return list of parameter arrays (weights, biases)
        return [self.weight, self.bias]
    def gradients(self):
        # Return list of gradient arrays corresponding to parameters
        return [self.grad_weight, self.grad_bias]
Insert your layer into a Network([...]) definition.

3. Implement a New Optimizer
Open optimizers.py.

Look at how SGD or Adam is implemented:

python
Kopieren
Bearbeiten
class MyCustomOptimizer:
    def __init__(self, parameters, lr=0.001, **kwargs):
        self.parameters = parameters  # List of parameter references
        self.lr = lr
        # Initialize any history/state here (e.g., momentum buffers)
     
    def step(self):
        # Iterate over all parameters and their stored gradients,
        # and update them according to your rule.
        for param, grad in self._gather_params_and_grads():
            # e.g., param -= self.lr * grad
            ...
Pass your optimizer to the Trainer in place of SGD or Adam.

📦 Installation & Usage
Clone & navigate

bash
Kopieren
Bearbeiten
git clone https://github.com/lennarddaw/Longshot.git
cd Longshot/src
Install dependencies

bash
Kopieren
Bearbeiten
pip install numpy
Run examples

Create a new Python script (example.py) in the repo root or src/ folder.

Copy the “Quick Example” code block into example.py.

Run:

bash
Kopieren
Bearbeiten
python example.py
Integrate into your project

Either add Longshot/src/ to your PYTHONPATH or install it as a local package:

bash
Kopieren
Bearbeiten
pip install -e .
Then import modules directly:

python
Kopieren
Bearbeiten
from longshot.activations import ReLU, Sigmoid
from longshot.layers import Dense
🛠️ Development & Testing
Run Unit Tests
(If you decide to add tests in a tests/ folder)

bash
Kopieren
Bearbeiten
pytest
Lint & Format
You can use flake8/black to ensure consistent style:

bash
Kopieren
Bearbeiten
pip install flake8 black
flake8 .
black .
🤝 Contributing
Contributions are welcome! To contribute:

Fork the repository.

Create a new branch (git checkout -b feature/my-feature).

Make your changes and add tests/examples if relevant.

Ensure all existing tests pass, and add new tests for your code.

Submit a pull request with a clear description of your changes.

Please review and adhere to any coding style guidelines. If there’s a CODE_OF_CONDUCT.md or CONTRIBUTING.md in the root, read it first.

📜 License
This project is licensed under the MIT License. See the LICENSE file for details.
