# =============================================================
# MNIST Digit Recognition using ANN (Keras + TensorFlow)
# =============================================================

import tensorflow
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


# ─────────────────────────────────────────────────────────────
# STEP 1: Load and Inspect Dataset
# ─────────────────────────────────────────────────────────────

(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

print("X_train shape:", X_train.shape)   # (60000, 28, 28)
print("X_test shape :", X_test.shape)    # (10000, 28, 28)
print("y_train sample:", y_train[:5])

plt.imshow(X_test[2])
plt.title(f"Label: {y_test[2]}")
plt.axis('off')
plt.show()


# ─────────────────────────────────────────────────────────────
# STEP 2: Preprocessing — Normalize Pixel Values
# ─────────────────────────────────────────────────────────────

X_train = X_train / 255.0    # scale from [0, 255] → [0.0, 1.0]
X_test  = X_test  / 255.0


# ─────────────────────────────────────────────────────────────
# STEP 3: Build the Model
# ─────────────────────────────────────────────────────────────

model = Sequential()

model.add(Flatten(input_shape=(28, 28)))     # 28×28 image → 784 flat vector
model.add(Dense(128, activation='relu'))     # hidden layer 1
model.add(Dense(32,  activation='relu'))     # hidden layer 2
model.add(Dense(10,  activation='softmax'))  # output: 10 digit classes (0–9)

model.summary()


# ─────────────────────────────────────────────────────────────
# STEP 4: Compile the Model
# ─────────────────────────────────────────────────────────────

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)


# ─────────────────────────────────────────────────────────────
# STEP 5: Train the Model
# ─────────────────────────────────────────────────────────────

history = model.fit(
    X_train, y_train,
    epochs=30,
    validation_split=0.2    # 20% of training data used for validation
)


# ─────────────────────────────────────────────────────────────
# STEP 6: Evaluate on Test Set
# ─────────────────────────────────────────────────────────────

y_prob = model.predict(X_test)          # shape: (10000, 10) — probabilities per class
y_pred = y_prob.argmax(axis=1)          # pick class with highest probability

print("Test Accuracy:", accuracy_score(y_test, y_pred))


# ─────────────────────────────────────────────────────────────
# STEP 7: Plot Training Curves
# ─────────────────────────────────────────────────────────────

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'],     label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'],     label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy Curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()


# ─────────────────────────────────────────────────────────────
# STEP 8: Predict a Single Image
# ─────────────────────────────────────────────────────────────

idx = 1000

plt.imshow(X_test[idx])
plt.title(f"True Label: {y_test[idx]}")
plt.axis('off')
plt.show()

predicted_digit = model.predict(X_test[idx].reshape(1, 28, 28)).argmax(axis=1)
print(f"Predicted digit: {predicted_digit[0]}")
