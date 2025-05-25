import numpy as np

def train_test_split(X, y, test_size=0.2, seed=None):
    """
    Teilt X und y in Trainings- und Testset auf.
    - test_size: Anteil der Daten im Testset (0–1).
    - seed: Zufallssamen für Reproduzierbarkeit.
    """
    if seed is not None:
        np.random.seed(seed)
    m = X.shape[1]
    # Shuffle-Index
    perm = np.random.permutation(m)
    X_shuffled = X[:, perm]
    y_shuffled = y[:, perm]
    # Split-Punkt
    split = int(m * (1 - test_size))
    X_train = X_shuffled[:, :split]
    X_test  = X_shuffled[:, split:]
    y_train = y_shuffled[:, :split]
    y_test  = y_shuffled[:, split:]
    return X_train, X_test, y_train, y_test

def create_batches(X, y, batch_size=32, shuffle=True):
    """
    Generator, der (X_batch, y_batch) in zufälliger Reihenfolge liefert.
    """
    m = X.shape[1]
    indices = np.arange(m)
    if shuffle:
        np.random.shuffle(indices)
    for start in range(0, m, batch_size):
        end = start + batch_size
        batch_idx = indices[start:end]
        yield X[:, batch_idx], y[:, batch_idx]

def normalize_minmax(X):
    """
    Skaliert X zeilenweise auf [0,1].
    """
    X_min = X.min(axis=1, keepdims=True)
    X_max = X.max(axis=1, keepdims=True)
    return (X - X_min) / (X_max - X_min + 1e-8)

def normalize_standard(X):
    """
    Z-standardisierung: (X - μ) / σ jeweils zeilenweise.
    """
    mu = X.mean(axis=1, keepdims=True)
    sigma = X.std(axis=1, keepdims=True)
    return (X - mu) / (sigma + 1e-8)

def one_hot_encode(y, num_classes):
    """
    Wandelt Vektor y (0..C-1) in One-Hot-Matrix um.
    Erwartet y Form (m,), gibt (C, m) zurück.
    """
    m = y.shape[0]
    one_hot = np.zeros((num_classes, m))
    one_hot[y, np.arange(m)] = 1
    return one_hot

def compute_accuracy(y_pred, y_true):
    """
    Berechnet Klassifikations-Accuracy.
    - y_pred: Vorhersage-Matrix (C, m), Wahrscheinlichkeiten.
    - y_true: One-Hot-Wahrheitsmatrix (C, m).
    """
    pred_labels = np.argmax(y_pred, axis=0)
    true_labels = np.argmax(y_true, axis=0)
    return np.mean(pred_labels == true_labels)
