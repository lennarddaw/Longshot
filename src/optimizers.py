class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, param, grad):
        return param - self.lr * grad
