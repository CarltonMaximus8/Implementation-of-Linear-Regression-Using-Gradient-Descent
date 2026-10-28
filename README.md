# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Start the program.
2. Import the required libraries and load the `50_Startups.csv` dataset.
3. Encode the categorical `State` column using one-hot encoding.
4. Separate the input features (X) and target variable (Profit).
5. Standardize the features and target variable using `StandardScaler`.
6. Add a bias term (column of ones) to the feature matrix.
7. Initialize the model parameters (theta) to zero.
8. Calculate the cost using Mean Squared Error.
9. Compute gradients and update theta using gradient descent until convergence or the maximum number of iterations is reached.
10. Plot the cost function to visualize convergence.
11. Prepare a new startup sample, encode and scale its features, and predict its profit.
12. Convert the predicted profit back to the original scale and display the result.
13. Stop the program.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Carlton Maximus A
RegisterNumber: 212225040052
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("50_Startups.csv")

print("First 5 rows of the dataset:")
display(data.head())
data_encoded = pd.get_dummies(data, columns=['State'], drop_first=True)

print("Encoded dataset:")
display(data_encoded.head())
X_raw = data_encoded.drop('Profit', axis=1).values

y_raw = data_encoded['Profit'].values.reshape(-1, 1)

print("Shape of features:", X_raw.shape)
print("Shape of target:", y_raw.shape)
X_scaler = StandardScaler()
y_scaler = StandardScaler()

X = X_scaler.fit_transform(X_raw)
y = y_scaler.fit_transform(y_raw)
m = X.shape[0]
X = np.hstack((np.ones((m, 1)), X))
print("Shape of X after adding bias term:", X.shape)
def compute_cost(X, y, theta):
    """
    Compute Mean Squared Error cost.
    """
    m = len(y)
    preds = X.dot(theta)
    cost = (1 / (2 * m)) * np.sum((preds - y) ** 2)
    return cost
def gradient_descent(X, y, learning_rate=0.01, num_iters=2000, tol=1e-8, verbose=False):
    """
    Batch Gradient Descent for Linear Regression.
    """
    m, n = X.shape
    theta = np.zeros((n, 1))
    J_history = []

    prev_cost = compute_cost(X, y, theta)

    for i in range(num_iters):
        preds = X.dot(theta)
        errors = preds - y
        grad = (1 / m) * (X.T.dot(errors))
        theta -= learning_rate * grad

        cost = compute_cost(X, y, theta)
        J_history.append(cost)

        if abs(prev_cost - cost) < tol:
            if verbose:
                print(f"Converged at iteration {i}")
            break
        prev_cost = cost

        if verbose and (i % 500 == 0 or i < 5):
            print(f"Iteration {i:4d}, Cost: {cost:.6f}")

    return theta, J_history
alpha = 0.01
theta, J_hist = gradient_descent(X, y, learning_rate=alpha, num_iters=5000, tol=1e-9, verbose=True)

print("\nLearned Parameters (Theta):")
print(theta.flatten())
plt.figure(figsize=(7,4))
plt.plot(J_hist)
plt.xlabel('Iterations')
plt.ylabel('Cost (MSE/2)')
plt.title('Cost Function Convergence')
plt.grid(True)
plt.show()
new_sample = pd.DataFrame([{
    'R&D Spend': 165349.2,
    'Administration': 136897.8,
    'Marketing Spend': 471784.1,
    'State': 'New York'
}])

new_encoded = pd.get_dummies(new_sample, columns=['State'], drop_first=True)

new_encoded = new_encoded.reindex(columns=data_encoded.drop('Profit', axis=1).columns, fill_value=0)

new_scaled = X_scaler.transform(new_encoded.to_numpy())
new_design = np.hstack((np.ones((new_scaled.shape[0], 1)), new_scaled))

scaled_pred = new_design.dot(theta)

pred_original = y_scaler.inverse_transform(scaled_pred)

print(f"\nPredicted Profit for the new startup: ₹{pred_original[0][0]:,.2f}")
 
*/
```

## Output:
![linear regression using gradient descent](sam.png)
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/16562e76-0358-4dcc-bfd3-611594f6669b" />
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/b82e5ae2-f39e-43fd-b1eb-8c33c2750dd7" />
<img width="868" height="437" alt="image" src="https://github.com/user-attachments/assets/166684af-41f6-436d-9d29-3949767c6af9" />
<img width="978" height="545" alt="image" src="https://github.com/user-attachments/assets/a5fa5b05-e1da-43f0-8db5-7de41304ff29" />
<img width="687" height="691" alt="image" src="https://github.com/user-attachments/assets/647f6877-8100-4ec7-a435-7e82fbea92d8" />


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
