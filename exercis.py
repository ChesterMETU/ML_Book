import numpy as np
import sklearn
import pandas as pd


# Perceptron
class Perceptron:
    def __init__(self, lr, randomState, nIter):
        self.lr = lr
        self.randomState = randomState
        self.nIter = nIter

    def fit(self, X, y):
        rgen = np.random.RandomState(self.randomState)
        self.w = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b = np.float64(0.0)
        self.errors = []
        for i in range(self.nIter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.lr * (target - self.predict(xi))
                self.w += update * xi
                self.b += update
                errors += int(update != 0.0)
            self.errors.append(errors)
        return self

    def net_input(self, X):
        return np.dot(X, self.w) + self.b

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, 0)


s = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

print("From URL:", s)

df = pd.read_csv(s, header=None, encoding="utf-8")
print(df.tail())

X = df.iloc[0:100, [0, 2]].values
y = df.iloc[0:100, 4].values
y = np.where(y == "Iris-setosa", 0, 1)

ppn = Perceptron(lr=0.01, nIter=50, randomState=2)

ppn.fit(X, y)


# Adaline
class AdalineGD:
    def __init__(self, lr, nIter, randomState):
        self.lr = lr
        self.nIter = nIter
        self.randomState = randomState

    def fit(self, X, y):
        rgen = np.random.RandomState(self.randomState)
        self.w = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b = np.float64(0.0)
        self.losses = []

        for _ in range(self.nIter):
            netInput = self.net_input(X)
            output = self.activation(netInput)
            errors = y - output
            self.w += self.lr * 2.0 * X.T.dot(errors) / X.shape[0]
            self.b += self.lr * 2.0 * errors.mean()
            loss = (errors**2).mean()
        self.losses.append(loss)

        return self

    def net_input(self, X):
        return np.dot(X, self.w) + self.b

    def activation(self, X):
        return X

    def predicted(self, X):
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)


adaline = AdalineGD(lr=0.01, nIter=50, randomState=2)

adaline.fit(X, y)

# Log Regression
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y
)

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

log = LogisticRegression(C=100, solver="lbfgs", multi_class="ovr")

log.fit(X_train_std, y_train)

print("LR Accuracy: %.3f" % log.score(X_test_std, y_test))
print(log.predict_proba(X_test_std[:3, :]))
print(log.predict_proba(X_test_std[:3, :]).argmax(axis=1))

# SVM
svc = SVC(kernel="rbf", gamma=0.10, C=10, random_state=1)

np.random.seed(1)
X_xor = np.random.randn(200, 2)
y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 0)

y_xor = np.where(y_xor, 1, 0)

svc.fit(X_xor, y_xor)

print(svc.score(X_xor, y_xor))
