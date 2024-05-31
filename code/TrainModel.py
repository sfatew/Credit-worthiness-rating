import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import ADASYN

def get_coefficient():

    # Load data
    file_path = 'Credit-worthiness-rating/data/german_credit.csv'
    data = np.loadtxt(file_path, delimiter=',', skiprows=1, dtype='int')

    X = data[:, 1:]
    y = data[:, 0]

    # ADASYN for X and y
    ada = ADASYN(random_state=42)
    X_res, y_res = ada.fit_resample(X, y)

    # initial model
    model = LogisticRegression(C = 1, max_iter = 5000, penalty = 'l2', solver = 'lbfgs')
    model.fit(X_res, y_res)

    # save the coeficient
    model_coeffictient = {
        "coefficients": model.coef_,
        "intercept":model.intercept_,
        "classes": model.classes_
    }
    return model_coeffictient

if __name__ == '__main__':
    import json
    model_coeficient = get_coefficient()
    model_coeficient["coefficients"] = model_coeficient["coefficients"].flatten().tolist()
    model_coeficient["intercept"] = model_coeficient["intercept"].flatten().tolist()
    model_coeficient["classes"] = model_coeficient["classes"].flatten().tolist()
    print(model_coeficient["coefficients"])
    print(model_coeficient["intercept"])
    print(model_coeficient["classes"])

    file_path = "Credit-worthiness-rating/model_coefficient.json"
    # Write data to JSON file
    with open(file_path, "w") as json_file:
        json.dump(model_coeficient, json_file)
