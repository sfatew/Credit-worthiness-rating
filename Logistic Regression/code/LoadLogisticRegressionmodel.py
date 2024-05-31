import json
from sklearn.linear_model import LogisticRegression
import numpy as np

def get_trained_model():
    model = LogisticRegression()

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

    return model
