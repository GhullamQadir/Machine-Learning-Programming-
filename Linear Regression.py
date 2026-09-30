# Linear Regression Code For Machine Learning

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures

# Generate synthetic data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)  # y = 4 + 3x + noise

# Train model
lin_reg = LinearRegression()
lin_reg.fit(X, y)

print(f"Intercept: {lin_reg.intercept_[0]:.4f}")
print(f"Coefficient: {lin_reg.coef_[0][0]:.4f}")

# Predictions
X_new = np.array([[0], [2]])
y_pred = lin_reg.predict(X_new)

# Evaluation
y_hat = lin_reg.predict(X)
print(f"R² Score: {r2_score(y, y_hat):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y, y_hat)):.4f}")
print(f"MAE: {mean_absolute_error(y, y_hat):.4f}")

# Visualization
plt.scatter(X, y, color='blue', alpha=0.5, label='Data')
plt.plot(X_new, y_pred, color='red', linewidth=2, label='Regression Line')
plt.xlabel('X'); plt.ylabel('y'); plt.legend(); plt.show()

# Polynomial Regression
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)
poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)
print(f"Poly R²: {r2_score(y, poly_reg.predict(X_poly)):.4f}")

# Regularization
ridge = Ridge(alpha=1.0)
ridge.fit(X, y)
lasso = Lasso(alpha=0.1)
lasso.fit(X, y)
elastic = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic.fit(X, y)
