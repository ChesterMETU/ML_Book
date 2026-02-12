from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
import numpy as np

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

print("Class labels: ", np.unique(y))
# print(X, y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y
)

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_Test_std = sc.transform(X_test)

ppn = Perceptron(eta0=0.1, random_state=1)
ppn.fit(X_train_std, y_train)

y_pred = ppn.predict(X_Test_std)
print("Misclassified examples: %d" % (y_test != y_pred).sum())
print(f"Total predictions {len(y_pred)}")

print("Accuracy: %.3f" % accuracy_score(y_test, y_pred))
print("Accuracy: %.3f" % ppn.score(X_Test_std, y_test))


class LogisticRegressionGD:
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
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = y - output
            self.w += self.lr * 2.0 * X.T.dot(errors) / X.shape[1]
            self.b += self.lr * 2.0 * errors.mean()

            loss = (
                -y.dot(np.log(output)) - ((1 - y).dot(np.log(1 - output))) / X.shape[1]
            )

            self.losses.append(loss)

        return self

    def net_input(self, X):
        return np.dot(X, self.w) + self.b

    def activation(self, z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -250, 250)))

    def predict(self, X):
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)


from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(C=100.0, solver="lbfgs", multi_class="ovr")

lr.fit(X_train_std, y_train)

print("Accuracy: %.3f" % lr.score(X_Test_std, y_test))

print(lr.predict_proba(X_Test_std[:3, :]))
print(lr.predict_proba(X_Test_std[:3, :]).argmax(axis=1))
