import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

# ── 1. Create Dataset ─────────────────────────────────────────────────
X, y = make_classification(
    n_samples=100,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    random_state=41
)

# Convert labels from {0,1} to {-1, +1}
y = np.where(y == 0, -1, 1)

# ── 2. Perceptron Class ───────────────────────────────────────────────
class Perceptron:

    def __init__(self, learning_rate=0.1, epochs=100):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Initialize weights and bias to zero
        self.weights = np.zeros(n_features)
        self.bias = 0

        for epoch in range(self.epochs):
            for idx, x_i in enumerate(X):

                # Calculate raw output
                linear_output = np.dot(x_i, self.weights) + self.bias

                # Predict using sign function
                y_pred = np.sign(linear_output)

                # Update only if misclassified
                if y_pred != y[idx]:
                    self.weights += self.lr * y[idx] * x_i
                    self.bias    += self.lr * y[idx]

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return np.sign(linear_output)

# ── 3. Train the Perceptron ───────────────────────────────────────────
p = Perceptron(learning_rate=0.1, epochs=100)
p.fit(X, y)

# ── 4. Accuracy ───────────────────────────────────────────────────────
y_pred = p.predict(X)
accuracy = np.mean(y_pred == y)
print(f"Accuracy: {accuracy * 100:.2f}%")

# ── 5. Plot Decision Boundary ─────────────────────────────────────────
def plot_decision_boundary(X, y, model):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))

    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid).reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3, cmap='bwr')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k')
    plt.title("Perceptron Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

plot_decision_boundary(X, y, p)
