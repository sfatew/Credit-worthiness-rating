import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score

# Load data
file_path = 'Credit-worthiness-rating/data/german_credit.csv'
data = np.loadtxt(file_path, delimiter=',', skiprows=1, dtype='int')

X = data[:, 1:]
y = data[:, 0]

# Initialize KFold
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Define the parameter grid
param_grid = {
    'solver': ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'],
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l2'],  # 'l1' or 'elasticnet' only supported by 'liblinear' and 'saga'
    'max_iter': [2000, 5000, 7500]
}

# Initialize the model
model = LogisticRegression()

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=skf, scoring='roc_auc', n_jobs=-1, verbose=2)

# Fit the model
grid_search.fit(X, y)

# Get the best parameters
best_params = grid_search.best_params_

# Evaluate the best model using cross-validation
best_model = LogisticRegression(**best_params)
score_kf_acc = cross_val_score(best_model, X, y, cv=skf, scoring='accuracy')
score_kf_f1 = cross_val_score(best_model, X, y, cv=skf, scoring='f1')
score_kf_auc = cross_val_score(best_model, X, y, cv=skf, scoring='roc_auc')
score_kf_precision = cross_val_score(best_model, X, y, cv=skf, scoring='precision')
score_kf_recall = cross_val_score(best_model, X, y, cv=skf, scoring='recall')

# Print the results
data = {
    "Precision": np.mean(score_kf_precision),
    "Recall": np.mean(score_kf_recall), 
    "Accuracy":np.mean(score_kf_acc), 
    "F1":np.mean(score_kf_f1), 
    "AUC":np.mean(score_kf_auc),
}

# Specify the file path
file_path = "Credit-worthiness-rating/LogisticRegression_result.json"

# Write data to JSON file
with open(file_path, "w") as json_file:
    json.dump(data, json_file)

