# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 1. Generate synthetic data
np.random.seed(42)
# X represents the independent variable
X = 2 * np.random.rand(100, 1)
# y represents the dependent variable (true equation: y = 4 + 3x + noise)
y = 4 + 3 * X + np.random.randn(100, 1)

# 2. Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 4. Make predictions using the test set
y_pred = model.predict(X_test)

# 5. Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Model Intercept: {model.intercept_[0]:.4f} (Expected: ~4)")
print(f"Model Coefficient: {model.coef_[0][0]:.4f} (Expected: ~3)")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R-squared Score: {r2:.4f}")

# 6. Visualize the results
plt.figure(figsize=(8, 6))

# Plot the actual testing data points
plt.scatter(X_test, y_test, color='black', label='Actual Data (Test Set)')

# Plot the predicted regression line
plt.plot(X_test, y_pred, color='blue', linewidth=3, label='Regression Line')

# Add labels and title
plt.xlabel('X (Independent Variable)')
plt.ylabel('y (Dependent Variable)')
plt.title('Simple Linear Regression')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# Display the plot
plt.show()
