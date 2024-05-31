import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from joblib import dump
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import ADASYN

# Load data
file_path = 'E:/University/Kì 2023.2/Machine Learning/Project/Credit-worthiness-rating/data/new_german_credit.csv'
data = np.loadtxt(file_path, delimiter=',', skiprows=1, dtype='int')

X = data[:, 1:]
y = data[:, 0]

# khởi tạo scaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Initialize KFold
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Initialize the model
model = LogisticRegression()

# Define the parameter grid
param_grid = {
    'solver': ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'],
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l2'],  # 'l1' or 'elasticnet' only supported by 'liblinear' and 'saga'
    'max_iter': [2000, 5000, 7500]
}
# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=skf, scoring='roc_auc', n_jobs=-1, verbose=2)
grid_search.fit(X, y)

# Fit the hyperparmeter optimize algorithm
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


conf_matrices = []
for i, (train, test) in enumerate(skf.split(X, y)):
    # Apply ADASYN to the training data
    ada = ADASYN(random_state=42)
    X_res, y_res = ada.fit_resample(X[train], y[train])
    
    model.fit(X_res, y_res)
    # Calculate confusion matrix for this fold
    y_pred = model.predict(X[test])
    conf_matrices.append(confusion_matrix(y[test], y_pred))

mean_conf_matrix = np.mean(conf_matrices, axis=0)

# Print the results
data = {
    "Precision": np.mean(score_kf_precision),
    "Recall": np.mean(score_kf_recall), 
    "Accuracy":np.mean(score_kf_acc), 
    "F1":np.mean(score_kf_f1), 
    "AUC":np.mean(score_kf_auc),
    'Confusion Matrix' : mean_conf_matrix.tolist()
}

##### Saving ######

# Specify the file path
file_path = "E:/University/Kì 2023.2/Machine Learning/Project/Credit-worthiness-rating/LogisticRegression_result.json"

# Write data to JSON file
with open(file_path, "w") as json_file:
    json.dump(data, json_file)

if __name__ == "__main__":
    print("best parameter: ", best_model)
