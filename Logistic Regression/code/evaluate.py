import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import json

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


model = LogisticRegression()


file_path = 'E:/University/Kì 2023.2/Machine Learning/Project/Credit-worthiness-rating/data/new_german_credit.csv'
data = np.loadtxt(file_path, delimiter=',', skiprows=1, dtype='int')

X = data[:, 1:]
y = data[:, 0]

# load hệ số cua model
# Mở tệp JSON
with open('E:/University/Kì 2023.2/Machine Learning/Project/Credit-worthiness-rating/model_coefficient.json', 'r') as f:
    coeficients_data = json.load(f)

coeficients_data["coefficients"] = np.array(coeficients_data["coefficients"]).reshape(1, len(coeficients_data["coefficients"]))
coeficients_data["intercept"] = np.array(coeficients_data["intercept"])
coeficients_data["classes"] = np.array(coeficients_data["classes"])


model.coef_ = coeficients_data['coefficients']
model.intercept_ = coeficients_data['intercept']
model.classes_ = coeficients_data['classes']

Z = model.predict_proba(X)

# probability to be 1 P(y =1 | x)
z = Z[:, 1]
# h is the sigmoid(z)
h = []
for z_i in z:
    h.append(sigmoid(z_i))


plt.plot(z, h, 'o')
plt.plot(z, y, 'o', color='r')
plt.axhline(0.5, color='b')
plt.xlabel('z value')
plt.ylabel('probability')
plt.show()
















