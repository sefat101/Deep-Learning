import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# ─────────────────────────────────────────────────────────────
# STEP 1: Load and Inspect Dataset
# ─────────────────────────────────────────────────────────────

(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

print("X_train shape:", X_train.shape)   # (60000, 28, 28)
print("X_test shape :", X_test.shape)    # (10000, 28, 28)
print("y_train sample:", y_train[:5])

plt.imshow(X_test[2], cmap='gray')
plt.title(f"Label: {y_test[2]}")
plt.axis('off')
plt.show()

# ─────────────────────────────────────────────────────────────
# STEP 2: Preprocessing — Normalize and reshape
# ─────────────────────────────────────────────────────────────

X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# ─────────────────────────────────────────────────────────────
# STEP 3: Build the CNN Model
# ─────────────────────────────────────────────────────────────

model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

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
# STEP 5: Define Callbacks
# ─────────────────────────────────────────────────────────────

# Stop training if validation loss doesn't improve
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

# Save best model to file
checkpoint = ModelCheckpoint(
    'best_mnist_cnn.h5',
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)

# Reduce learning rate if validation loss plateaus
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

callbacks = [early_stop, checkpoint, reduce_lr]

# ─────────────────────────────────────────────────────────────
# STEP 6: Train the Model
# ─────────────────────────────────────────────────────────────

history = model.fit(
    X_train, y_train,
    epochs=30,
    validation_split=0.2,
    batch_size=128,
    callbacks=callbacks
)

# ─────────────────────────────────────────────────────────────
# STEP 7: Evaluate on Test Set
# ─────────────────────────────────────────────────────────────

y_prob = model.predict(X_test)
y_pred = y_prob.argmax(axis=1)

print("Test Accuracy:", accuracy_score(y_test, y_pred))

# ─────────────────────────────────────────────────────────────
# STEP 8: Plot Training Curves
# ─────────────────────────────────────────────────────────────

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy Curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# ─────────────────────────────────────────────────────────────
# STEP 9: Predict a Single Image
# ─────────────────────────────────────────────────────────────

idx = 1000

plt.imshow(X_test[idx].reshape(28, 28), cmap='gray')
plt.title(f"True Label: {y_test[idx]}")
plt.axis('off')
plt.show()

predicted_digit = model.predict(X_test[idx].reshape(1, 28, 28, 1)).argmax(axis=1)
print(f"Predicted digit: {predicted_digit[0]}")
