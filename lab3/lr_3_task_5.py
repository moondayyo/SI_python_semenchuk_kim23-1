# Варіант 3 (за списком 13)

import numpy as np
from sklearn import linear_model
import sklearn.metrics as sm
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

np.random.seed(0)
m = 100
X = 6 * np.random.rand(m, 1) - 4
y = 0.5 * X ** 2 + X + 2 + np.random.randn(m, 1)

# Побудова графіка
plt.scatter(X, y, color='green', label='data')

# Лінійна регресія
linear_regression = linear_model.LinearRegression()
linear_regression.fit(X, y)
y_linear_pred = linear_regression.predict(X)

# Поліноміальна регресія
polynomial_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = polynomial_features.fit_transform(X)

linear_regression_poly = linear_model.LinearRegression()
linear_regression_poly.fit(X_poly, y)

y_poly_pred = linear_regression_poly.predict(X_poly)

# Побудова графіка
plt.plot(X, y_linear_pred, color='blue', label='linear', linewidth=4)
sort_indices = np.argsort(X[:, 0])
X_sorted = X[sort_indices]
y_poly_pred_sorted = y_poly_pred[sort_indices]
plt.plot(X_sorted, y_poly_pred_sorted, color='red', label='polynomial', linewidth=4)
plt.xticks(())
plt.yticks(())
plt.show()

print("Linear regressor performance:")
print("Mean absolute error =",
round(sm.mean_absolute_error(y, y_linear_pred), 2))
print("Mean squared error =",
round(sm.mean_squared_error(y, y_linear_pred), 2))
print("Median absolute error =",
round(sm.median_absolute_error(y, y_linear_pred), 2))
print("Explain variance score =", round(sm.explained_variance_score(y, y_linear_pred), 2))
print("R2 score =", round(sm.r2_score(y, y_linear_pred), 2))

print("\nPolinomial regressor performance:")
print("Mean absolute error =",
round(sm.mean_absolute_error(y, y_poly_pred), 2))
print("Mean squared error =",
round(sm.mean_squared_error(y, y_poly_pred), 2))
print("Median absolute error =",
round(sm.median_absolute_error(y, y_poly_pred), 2))
print("Explain variance score =", round(sm.explained_variance_score(y, y_poly_pred), 2))
print("R2 score =", round(sm.r2_score(y, y_poly_pred), 2))

print('\n', linear_regression_poly.coef_, linear_regression_poly.intercept_)