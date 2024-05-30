import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score, precision_score
from xgboost import XGBClassifier
from scipy.stats import uniform, randint
from sklearn.model_selection import RandomizedSearchCV
from imblearn.over_sampling import SMOTE
import warnings
import pickle

warnings.filterwarnings('ignore')
plt.style.use('ggplot')

def xgboost_model():
    # Load the dataset

    df = pd.read_csv('new_german_credit_discrete.csv')
    df.drop(columns=['Unnamed: 0'], inplace=True)  # Drop unnecessary index column
    df.head()
    df.describe()
    df.info()

    # Correlation matrix

    plt.figure(figsize=(18, 12))
    corr = df.corr()
    sns.heatmap(corr, annot=True, fmt='.2g', linewidths=1)
    plt.show()

    # Data preprocessing

    X = df.drop(['Creditability'], axis=1)
    y = df['Creditability']

    # Label encoding

    label_encoder = LabelEncoder()
    for column in X.columns:
        if X[column].dtype == 'object':
            X[column] = label_encoder.fit_transform(X[column])

    # Handling class imbalance

    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)

    # Visualizing distribution of variables 

    plt.figure(figsize=(25, 20))
    plotnumber = 1
    for col in X.columns:
        if plotnumber <= 24:
            ax = plt.subplot(5, 5, plotnumber)
            sns.distplot(X[col])
            plt.xlabel(col, fontsize=15)
        plotnumber += 1
    plt.tight_layout()
    plt.show()

    # Initialize XGBoost classifier

    xgbc = XGBClassifier(n_estimators=120, use_label_encoder=False, eval_metric='logloss')

    # Hyperparameter tuning with RandomizedSearchCV

    param_dist = {
        'learning_rate': uniform(0.01, 0.1),
        'max_depth': randint(1, 10),
        'n_estimators': randint(10, 500),
        'gamma': uniform(0.01, 0.1),
        'subsample': uniform(0.1, 0.5),
        'colsample_bytree': uniform(0.1, 0.5)
    }

    xgbc_rscv = RandomizedSearchCV(
        estimator=xgbc,
        param_distributions=param_dist,
        n_iter=100,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )

    xgbc_rscv.fit(X_res, y_res)

    best_params = xgbc_rscv.best_params_
    model = XGBClassifier(**best_params, use_label_encoder=False, eval_metric='logloss')
    model.fit(X_res, y_res)

    # k-fold cross validation

    from sklearn.model_selection import StratifiedKFold, cross_val_score


    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    mean_accuracy = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')

    print(f'K-Fold CV Accuracy: {mean_accuracy.mean()} ± {mean_accuracy.std()}')

    mean_f1 = cross_val_score(model, X, y, cv=kfold, scoring='f1')

    print(f'K-Fold CV F1 Score: {mean_f1.mean()} ± {mean_f1.std()}')

    mean_auc = cross_val_score(model, X, y, cv=kfold, scoring='roc_auc')

    print(f'K-Fold CV AUC: {mean_auc.mean()} ± {mean_auc.std()}')

    mean_recall = cross_val_score(model, X, y, cv=kfold, scoring='recall')

    print(f'K-Fold CV Recall: {mean_recall.mean()} ± {mean_recall.std()}')

    mean_precision = cross_val_score(model, X, y, cv=kfold, scoring='precision')

    print(f'K-Fold CV Precision: {mean_precision.mean()} ± {mean_precision.std()}')

    return model


