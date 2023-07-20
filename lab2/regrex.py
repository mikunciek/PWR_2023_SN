import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

#Provide data
x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))  #regressors
y = np.array([5, 20, 14, 32, 22, 38]) #response

#Create model
model = LinearRegression().fit(x, y)

#Results
r_sq = model.score(x, y) #R^2
print(f"coefficient of determination: {r_sq}")
print(f"intercept: {model.intercept_}") #b0
print(f"slope: {model.coef_}") #b1


#Predict response
y_pred = model.predict(x)
print(f"predicted response:\n{y_pred}")
y_pred = model.intercept_ + model.coef_ * x
print(f"predicted response:\n{y_pred}")

#Plot output
plt.scatter(x, y, color='red')
plt.plot(x, y_pred, color='blue', linewidth=1)
plt.title('Linear Regression')
plt.show()


# Load the diabetes dataset
#diabetes = datasets.load_diabetes()
#f, axarr = plt.subplots(5, 2, sharex=True, sharey=True, figsize=(12, 12))
# for i in range(0, 5):
#     for j in range(0, 2):
#         # Use only one feature
#         diabetes_X = diabetes.data[:, np.newaxis, i * 2 + j]
#         # Split the data into training/testing sets
#         diabetes_X_train = diabetes_X[:-20]
#         diabetes_X_test = diabetes_X[-20:]
#         # Split the targets into training/testing sets
#         diabetes_y_train = diabetes.target[:-20]
#         diabetes_y_test = diabetes.target[-20:]
#         # Create linear regression object
#         regr = linear_model.LinearRegression()
#         # Train the model using the training sets
#         regr.fit(diabetes_X_train, diabetes_y_train)
#         # The coefficients
#         print('Coefficients: \n', regr.coef_)
#         # The mean square error
#         print("Residual sum of squares: %.2f"
#               % np.mean((regr.predict(diabetes_X_test) - diabetes_y_test) ** 2))
#         # Explained variance score: 1 is perfect prediction
#         print('Variance score: %.2f' % regr.score(diabetes_X_test, diabetes_y_test))
#         # Plot outputs
#         axarr[i, j].scatter(diabetes_X_test, diabetes_y_test, color='red')
#         axarr[i, j].plot(diabetes_X_test, regr.predict(diabetes_X_test), color='blue', linewidth=1)
# plt.show()
#

