import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_boston

df = df[['text', 'label_num']]
print("✅ Dataset Loaded Successfully!")
print(df.head())

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
print("Linear Regression R²:", r2_score(y_test, y_pred_lr))
print("Linear Regression RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_lr)))

# 2. Ridge Regression
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)
print("Ridge Regression R²:", r2_score(y_test, y_pred_ridge))
print("Ridge Regression RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_ridge)))

# 3. Polynomial Regression (Degree 2)
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)
poly_reg = LinearRegression()
poly_reg.fit(X_train_poly, y_train)
y_pred_poly = poly_reg.predict(X_test_poly)
print("Polynomial Regression R²:", r2_score(y_test, y_pred_poly))
print("Polynomial Regression RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_poly)))

# Optional: Visualize predictions for a single feature (e.g., 'CRIM')
plt.figure(figsize=(10, 6))
plt.scatter(X_test['CRIM'], y_test, color='blue', label='Actual')
plt.scatter(X_test['CRIM'], y_pred_lr, color='red', label='Linear Pred')
plt.scatter(X_test['CRIM'], y_pred_ridge, color='green', label='Ridge Pred')
plt.scatter(X_test['CRIM'], y_pred_poly, color='orange', label='Poly Pred')
plt.xlabel('CRIM (Crime Rate)')
plt.ylabel('Price')
plt.legend()
plt.title('House Price Prediction Comparison')
plt.show()
