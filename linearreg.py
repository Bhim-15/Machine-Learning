
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


X = np.array([[1], [2], [3], [4], [5]])  
y = np.array([10, 20, 30, 40, 50])      


model = LinearRegression()
model.fit(X, y)


X_new = np.array([[6]])  
y_pred = model.predict(X_new)

print(f"Predicted score for 6 hours of study: {y_pred[0]:.2f}")


plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, model.predict(X), color='red', label='Regression Line')
plt.xlabel("Hours Studied")
plt.ylabel("Score Obtained")
plt.title("Simple Linear Regression")
plt.legend()
plt.grid(True)
plt.show()
