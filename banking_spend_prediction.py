import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

X = np.array([30,40,50,60,70,80,90,100,110,120]).reshape(-1,1)
y = np.array([10,12,15,18,20,25,28,30,35,40])

model = LinearRegression()
model.fit(X, y)

slope = model.coef_[0]
intercept = model.intercept_

print("Slope:", slope)
print("Intercept:", intercept)

new_income = np.array([85, 90, 26, 200, 10]).reshape(-1,1)
predictions = model.predict(new_income)

print("Predictions:", predictions)

plt.scatter(X, y)
plt.plot(X, model.predict(X))
plt.scatter(new_income, predictions)
plt.xlabel("Income")
plt.ylabel("Spend")
plt.title("Banking Spend Prediction")
plt.show()