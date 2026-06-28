import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

# Dataset
X, y = make_classification(
    n_samples=100,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    random_state=42
)

# Convert labels to {-1,+1}
y = np.where(y == 0, -1, 1)

# Initialize weights and bias
w = np.zeros(X.shape[1])
b = 0

epochs = 100

for epoch in range(epochs):

    for i in range(len(X)):

        x_i = X[i]

        # Prediction
        z = np.dot(x_i, w) + b

        y_pred = 1 if z >= 0 else -1

        # Misclassified?
        if y_pred != y[i]:

            # Perceptron Trick
            w = w + y[i] * x_i
            b = b + y[i]

print("Weights:", w)
print("Bias:", b)
