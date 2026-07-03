# ==============================================================================================================================
# CUSTOMER CHURN PREDICTION USING ANN
# Dataset - https://www.kaggle.com/datasets/rjmanoj/credit-card-customer-churn-prediction
# ==============================================================================================================================


# Import Libraries
import numpy as np
import pandas as pd
import tensorflow as tf

# ------------------------------------------
# Load Dataset
# ------------------------------------------
df = pd.read_csv('Churn_Modelling.csv')

# ------------------------------------------
# Remove Unnecessary Columns
# ------------------------------------------
df.drop(columns=['RowNumber', 'CustomerId', 'Surname'], inplace=True)

# ------------------------------------------
# Separate Features and Target
# ------------------------------------------
X = df.drop(columns=['Exited'])
y = df['Exited']

# ------------------------------------------
# Label Encoding (Gender)
# ------------------------------------------
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
X['Gender'] = le.fit_transform(X['Gender'])

# ------------------------------------------
# One Hot Encoding (Geography)
# ------------------------------------------
X = pd.get_dummies(X, columns=['Geography'], drop_first=True)

# Convert boolean columns to int
X = X.astype(int)

# ------------------------------------------
# Train Test Split
# ------------------------------------------
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ------------------------------------------
# Feature Scaling
# ------------------------------------------
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ------------------------------------------
# Build ANN Model
# ------------------------------------------
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()

# Hidden Layer 1
model.add(Dense(
    units=11,
    activation='relu',
    input_dim=X_train.shape[1]
))

# Hidden Layer 2
model.add(Dense(
    units=11,
    activation='relu'
))

# Output Layer
model.add(Dense(
    units=1,
    activation='sigmoid'
))

# ------------------------------------------
# Model Summary
# ------------------------------------------
model.summary()

# ------------------------------------------
# Compile Model
# ------------------------------------------
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ------------------------------------------
# Train Model
# ------------------------------------------
history = model.fit(
    X_train,
    y_train,
    batch_size=32,
    epochs=100,
    validation_split=0.2,
    verbose=1
)

# ------------------------------------------
# Predict on Test Data
# ------------------------------------------
y_pred = model.predict(X_test)

# Convert probabilities into classes
y_pred = (y_pred > 0.5)

# ------------------------------------------
# Evaluation
# ------------------------------------------
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

print("\nAccuracy:")
print(accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ------------------------------------------
# Plot Accuracy Curve
# ------------------------------------------
import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'])
plt.show()

# ------------------------------------------
# Plot Loss Curve
# ------------------------------------------
plt.figure(figsize=(8,5))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'])
plt.show()

# ------------------------------------------
# Predict for One New Customer
# ------------------------------------------

sample = np.array([[600,      # CreditScore
                    1,        # Gender (Male=1)
                    40,       # Age
                    3,        # Tenure
                    60000,    # Balance
                    2,        # NumOfProducts
                    1,        # HasCrCard
                    1,        # IsActiveMember
                    50000,    # EstimatedSalary
                    0,        # Germany
                    0]])      # Spain

sample = scaler.transform(sample)

prediction = model.predict(sample)

print("\nChurn Probability:", prediction[0][0])

if prediction > 0.5:
    print("Customer Will Leave")
else:
    print("Customer Will Stay")
