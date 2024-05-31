import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import ADASYN

# Load data
file_path = 'Credit-worthiness-rating/data/german_credit.csv'
data = np.loadtxt(file_path, delimiter=',', skiprows=1, dtype='int')

X = data[:, 1:]
y = data[:, 0]

# Initialize KFold
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

# Initialize model
model = LogisticRegression(C=1, max_iter=5000, penalty='l2', solver='lbfgs')

# Lists to store the false positive rates, true positive rates, aucs, and confusion matrices
tprs = []
aucs = []
conf_matrices = []
mean_fpr = np.linspace(0, 1, 100000)

# Create a figure for the ROC plot
plt.figure(figsize=(10, 8))

for i, (train, test) in enumerate(kf.split(X, y)):
    # Apply ADASYN to the training data
    ada = ADASYN(random_state=42)
    X_res, y_res = ada.fit_resample(X[train], y[train])
    
    model.fit(X_res, y_res)
    probas_ = model.predict_proba(X[test])
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    tprs.append(np.interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3, label=f'ROC fold {i+1} (AUC = {roc_auc:.2f})')

    # Calculate confusion matrix for this fold
    y_pred = model.predict(X[test])
    conf_matrices.append(confusion_matrix(y[test], y_pred))

# Print the mean ROC AUC score
score_kf_auc = cross_val_score(model, X, y, cv=kf, scoring='roc_auc')
print("The mean roc_auc score: ", np.mean(score_kf_auc))

# Plot the chance line
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=0.8)

# Plot the mean ROC curve
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=0.8)

# Fill the area between the mean ROC and the standard deviation
std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.2, label=r'$\pm$ 1 std. dev.')

# Plot details
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Calculate mean confusion matrix
mean_conf_matrix = np.mean(conf_matrices, axis=0)

# Plot mean confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=mean_conf_matrix, display_labels=model.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title('Mean Confusion Matrix')
plt.show()
