import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# 1. Generate a larger synthetic dataset for better training
np.random.seed(42)
X = 2 * np.random.rand(1000, 1)
# Adding slightly more noise to make regularization useful
y = 4 + 3 * X.ravel() + np.random.randn(1000) * 1.5 

# 2. Initial Split: 80% for Training/Validation, 20% for Final Testing
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Build a Pipeline combining a Scaler and the SGD Model
# We use SGDRegressor to allow for iterative training, callbacks (early stopping), and regularization
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', SGDRegressor(
        early_stopping=True,        # "Callback": stops training when validation score stops improving
        validation_fraction=0.2,    # Dedicates 20% of training data for the early stopping validation
        n_iter_no_change=5,         # "Patience": stops after 5 epochs with no improvement
        tol=1e-3,                   # Tolerance for improvement
        max_iter=1000,              # Maximum epochs
        random_state=42
    ))
])

# 4. Define the Hyperparameter Grid
param_grid = {
    'model__penalty': ['l2', 'l1', 'elasticnet'],  # Regularization types
    'model__alpha': [0.0001, 0.001, 0.01, 0.1],    # Regularization strength
    'model__learning_rate': ['constant', 'invscaling', 'adaptive'], # How step size changes
    'model__eta0': [0.01, 0.1]                     # Initial learning rate
}

# 5. Initialize GridSearchCV for Hyperparameter Tuning with 5-Fold Cross Validation
print("Starting Hyperparameter Tuning...")
search = GridSearchCV(
    pipeline, 
    param_grid, 
    cv=5,                                  # 5-fold cross-validation
    scoring='neg_mean_squared_error',      # Optimize to minimize MSE
    n_jobs=-1,                             # Use all available CPU cores
    verbose=1
)

# Fit the grid search to the training data
search.fit(X_temp, y_temp)

# Extract the best model from the search
best_model = search.best_estimator_

print("\n--- Tuning Complete ---")
print(f"Best Hyperparameters: {search.best_params_}")

# 6. Evaluate the Best Model on the Unseen Test Set
y_pred = best_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Test Set Performance ---")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R-squared Score: {r2:.4f}")

# 7. Visualize the Results
plt.figure(figsize=(10, 6))

# Plot actual test data
plt.scatter(X_test, y_test, color='black', alpha=0.5, label='Actual Data (Test Set)')

# Plot regression line
# We create a line from the min to max X values to ensure a smooth, continuous line
X_line = np.linspace(X_test.min(), X_test.max(), 100).reshape(-1, 1)
y_line = best_model.predict(X_line)
plt.plot(X_line, y_line, color='blue', linewidth=3, label='Optimized Regression Line')

plt.xlabel('X (Scaled Iteratively)')
plt.ylabel('y')
plt.title('Tuned SGD Linear Regression with Early Stopping')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()
