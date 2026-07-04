# =====================================================
# MANUAL BACKPROPAGATION (2-2-1 NETWORK)
# NO KERAS / TENSORFLOW
# ONLY NUMPY
# =====================================================

import numpy as np
import pandas as pd

# =====================================================
# DATASET
# =====================================================

df = pd.DataFrame(
    [
        [8, 8, 4],
        [7, 9, 5],
        [6,10, 6],
        [5,12, 7]
    ],
    columns=['cgpa','profile_score','lpa']
)

print("Dataset:")
print(df)

# =====================================================
# PICK FIRST STUDENT
# =====================================================

X = df[['cgpa','profile_score']].values[0].reshape(2,1)

y = df['lpa'].values[0]

print("\nInput X:")
print(X)

print("\nActual Output y:")
print(y)

# =====================================================
# INITIALIZE PARAMETERS
# =====================================================

def initialize_parameters():

    parameters = {

        'W1': np.array([
            [0.1,0.1],
            [0.1,0.1]
        ]),

        'b1': np.zeros((2,1)),

        'W2': np.array([
            [0.1],
            [0.1]
        ]),

        'b2': np.zeros((1,1))
    }

    return parameters

# =====================================================
# FORWARD PROPAGATION
# =====================================================

def forward(X, parameters):

    W1 = parameters['W1']
    b1 = parameters['b1']

    W2 = parameters['W2']
    b2 = parameters['b2']

    # Hidden Layer

    Z1 = np.dot(W1.T, X) + b1

    A1 = Z1

    # Output Layer

    Z2 = np.dot(W2.T, A1) + b2

    y_hat = Z2

    cache = {
        'X': X,
        'Z1': Z1,
        'A1': A1,
        'Z2': Z2
    }

    return y_hat, cache

# =====================================================
# LOSS FUNCTION
# =====================================================

def compute_loss(y, y_hat):

    return (y - y_hat[0][0])**2

# =====================================================
# BACKPROPAGATION
# =====================================================

def backward(parameters, cache, y, y_hat):

    X = cache['X']
    A1 = cache['A1']

    # -------------------------
    # OUTPUT LAYER
    # -------------------------

    dL_dyhat = -2 * (y - y_hat)

    dW2 = A1 * dL_dyhat

    db2 = dL_dyhat

    # -------------------------
    # HIDDEN LAYER
    # -------------------------

    dA1 = parameters['W2'] * dL_dyhat

    dW1 = np.dot(X, dA1.T)

    db1 = dA1

    grads = {

        'dW1': dW1,
        'db1': db1,

        'dW2': dW2,
        'db2': db2
    }

    return grads

# =====================================================
# UPDATE PARAMETERS
# =====================================================

def update_parameters(parameters, grads, lr):

    parameters['W1'] -= lr * grads['dW1']
    parameters['b1'] -= lr * grads['db1']

    parameters['W2'] -= lr * grads['dW2']
    parameters['b2'] -= lr * grads['db2']

    return parameters

# =====================================================
# TRAINING
# =====================================================

parameters = initialize_parameters()

epochs = 20

learning_rate = 0.001

print("\n================ TRAINING ================\n")

for epoch in range(epochs):

    # Forward Pass
    y_hat, cache = forward(X, parameters)

    # Loss
    loss = compute_loss(y, y_hat)

    # Backward Pass
    grads = backward(
        parameters,
        cache,
        y,
        y_hat
    )

    # Update Parameters
    parameters = update_parameters(
        parameters,
        grads,
        learning_rate
    )

    print(
        f"Epoch {epoch+1:02d} | "
        f"Prediction = {y_hat[0][0]:.4f} | "
        f"Loss = {loss:.4f}"
    )

# =====================================================
# FINAL PARAMETERS
# =====================================================

print("\n================ FINAL PARAMETERS ================\n")

for key, value in parameters.items():

    print(key)
    print(value)
    print()

# =====================================================
# FINAL PREDICTION
# =====================================================

y_hat, _ = forward(X, parameters)

print("Final Prediction:", y_hat[0][0])
print("Actual Value:", y)




#you can execute this code block by block and see what happens for each student after updates params through backpropagation 
