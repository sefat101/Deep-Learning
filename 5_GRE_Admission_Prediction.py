# ==========================================
# GRADUATE ADMISSION PREDICTION USING ANN
# OPTIMIZED VERSION
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# ==========================================
# REPRODUCIBILITY
# ==========================================

np.random.seed(42)
tf.random.set_seed(42)

# ==========================================
# LOAD & CLEAN DATASET
# ==========================================

df = pd.read_csv("Admission_Predict_Ver1.1.csv")
df.drop(columns=['Serial No.'], inplace=True)

# Strip whitespace from column names (fixes 'Chance of Admit ' bug)
df.columns = df.columns.str.strip()

print("Shape:", df.shape)
print(df.describe())
print("Missing values:\n", df.isnull().sum())

# ==========================================
# INPUTS AND OUTPUT
# ==========================================

X = df.drop(columns=['Chance of Admit'])
y = df['Chance of Admit']

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# ==========================================
# FEATURE SCALING
# ==========================================

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ==========================================
# BUILD OPTIMIZED ANN MODEL
# ==========================================

model = Sequential([

    # Hidden Layer 1 — more neurons + batch norm
    Dense(64, activation='relu', input_dim=X_train.shape[1]),
    BatchNormalization(),
    Dropout(0.2),

    # Hidden Layer 2
    Dense(32, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),

    # Hidden Layer 3
    Dense(16, activation='relu'),

    # Output — linear for regression
    Dense(1, activation='linear')
])

# ==========================================
# COMPILE
# ==========================================

model.compile(
    loss='mean_squared_error',
    optimizer=Adam(learning_rate=0.001),
    metrics=['mae']
)

model.summary()

# ==========================================
# CALLBACKS
# ==========================================

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=15,           # stop if no improvement for 15 epochs
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,            # halve the learning rate
    patience=7,
    min_lr=1e-6,
    verbose=1
)

# ==========================================
# TRAIN MODEL
# ==========================================

history = model.fit(
    X_train_scaled, y_train,
    epochs=500,            # high max — early stopping will cut it short
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# ==========================================
# EVALUATION
# ==========================================

y_pred = model.predict(X_test_scaled).flatten()

r2  = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("\n========== MODEL PERFORMANCE ==========")
print(f"R² Score : {r2:.4f}")
print(f"MAE      : {mae:.4f}")
print(f"MSE      : {mse:.4f}")
print(f"RMSE     : {rmse:.4f}")

# ==========================================
# LOSS CURVES
# ==========================================

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'],     label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'],     label='Train MAE')
plt.plot(history.history['val_mae'], label='Val MAE')
plt.title("MAE Curve")
plt.xlabel("Epochs")
plt.ylabel("MAE")
plt.legend()

plt.tight_layout()
plt.show()

# ==========================================
# ACTUAL vs PREDICTED PLOT
# ==========================================

plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color='steelblue', edgecolors='k')
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Fit')
plt.title("Actual vs Predicted")
plt.xlabel("Actual Chance of Admit")
plt.ylabel("Predicted Chance of Admit")
plt.legend()
plt.tight_layout()
plt.show()

# ==========================================
# NEW STUDENT PREDICTION
# ==========================================

student = np.array([[
    337,   # GRE Score
    118,   # TOEFL Score
    4,     # University Rating
    4.5,   # SOP
    4.5,   # LOR
    9.65,  # CGPA
    1      # Research (1=Yes, 0=No)
]])

student_scaled = scaler.transform(student)
prediction = model.predict(student_scaled)

print(f"\nChance of Admit = {prediction[0][0]:.4f}  ({prediction[0][0]*100:.2f}%)")
