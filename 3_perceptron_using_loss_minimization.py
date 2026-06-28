import numpy as np
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt


# Dataset
X, y = make_classification(
    n_samples=100,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    random_state=42
)

y = np.where(y == 0, -1, 1)

# Initialize
w = np.zeros(X.shape[1])
b = 0

lr = 0.01
epochs = 100

loss_history = []

for epoch in range(epochs):

    total_loss = 0

    for i in range(len(X)):

        margin = y[i] * (np.dot(X[i], w) + b)

        if margin <= 0:

            # Gradient update
            w += lr * y[i] * X[i]
            b += lr * y[i]

            total_loss += -margin

    loss_history.append(total_loss)

print("Weights:", w)
print("Bias:", b)
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Perceptron Loss")
plt.title("Training Loss")
plt.show()
