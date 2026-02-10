import numpy as np
import pandas as pd


class AdalineGD:
    def __init__(self, lr=0.01, nIter=50, randomState=1):
        self.lr = lr
        self.nIter = nIter
        self.randomState = randomState

    def fit(self, X, y):
        rgen = np.random.RandomState(self.randomState)
        self.w = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])

        self.b = np.float64(0.0)
        self.losses = []

        for i in range(self.nIter):
            netInput = self.net_input(X)
            output = self.activation(netInput)
            errors = y - output
            self.w = self.lr * 2.0 * X.T.dot(errors) / X.shape[0]
            self.b = self.lr * 2.0 * errors.mean()
            loss = (errors**2).mean()
            self.losses.append(loss)
            print(X.shape)
            print(errors.shape)

        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w) + self.b

    def activation(self, X):
        """Compute linear activation"""
        return X

    def predict(self, X):
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)


s = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

print("From URL:", s)

df = pd.read_csv(s, header=None, encoding="utf-8")
print(df.tail())

# select setosa and versicolor
y = df.iloc[0:100, 4].values
y = np.where(y == "Iris-setosa", 0, 1)
# extract sepal length and petal length
X = df.iloc[0:100, [0, 2]].values

print(X)
adaline = AdalineGD()

adaline.fit(X, y)


class AdalineSGD:
    def __init__(self, lr=0.01, nIter=50, shuffle=True, randomState=None):
        self.lr = lr
        self.nIter = nIter
        self.shuffle = shuffle
        self.randomState = randomState

    def fit(self, X, y):
        self.initializeWeights(X.shape[1])
        self.losses = []
        for i in range(self.nIter):
            if self.shuffle:
                X, y = self.shuffle(X, y)
            losses = []
            for xi, target in zip(X, y):
                losses.append(self.updateWeights(xi, target))
            avgLoss = np.mean(losses)
            self.losses.append(losses)
        return self

    def partial_fit(self, X, y):
        """Fit training data without reinitializing the weights"""
        if not self.w_initialized:
            self.initializeWeights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self.updateWeights(xi, target)
        else:
            self.updatWeights(X, y)
        return self

    def initializeWeights(self, m):
        self.rgen = np.random.RandomState(self.randomState)
        self.w = self.rgen.normal(loc=0.0, scale=0.01, size=m)
        self.b = np.float64(0.0)
        self = wInitialized = True

    def shuffle(self, X, y):
        r = self.rgen.permutation(len(y))
        return X[r], y[r]

    def updateWeights(self, xi, target):
        output = self.activation(self.net_input(xi))
        error = target - output
        self.w += self.lr * 2.0 * xi * error
        self.b += self.lr * 2.0 * error
        loss = error**2
        return loss

    def net_input(self, X):
        return np.dot(X, self.w) + self.b

    def activation(self, X):
        return X

    def predict(self, X):
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)
