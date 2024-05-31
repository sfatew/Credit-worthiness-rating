import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, recall_score, roc_curve, roc_auc_score, precision_score, precision_recall_curve
from xgboost import XGBClassifier
from scipy.stats import uniform, randint
from sklearn.model_selection import RandomizedSearchCV
from imblearn.over_sampling import ADASYN
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

    # Apply scaling
   
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Save the scaler
    joblib.dump(scaler, 'scaler.joblib')

    # Handling class imbalance

    adas = ADASYN(random_state=42)

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

    # Initialize outer and inner K-fold cross-validation
    outer_kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
    inner_kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
    
    # Define parameter grid for hyperparameter tuning
    param_dist = {
        'learning_rate': uniform(0.01, 0.1),
        'max_depth': randint(1, 10),
        'n_estimators': randint(10, 500),
        'gamma': uniform(0.01, 0.1),
        'subsample': uniform(0.1, 0.5),
        'colsample_bytree': uniform(0.1, 0.5),
        'scale_pos_weight': uniform(2.30,2.36),
        'min_child_weight': [5, 10, 15, 20,25]
    }

    best_params_list = []
    y_true = []
    y_pred = []
    y_pred_proba = []

    for train_val_index, test_index in outer_kf.split(X_scaled, y):
            X_train_val, X_test = X_scaled[train_val_index], X_scaled[test_index]
            y_train_val, y_test = y[train_val_index], y[test_index]

            # Apply ADASYN for the training data
            X_train_res, y_train_res = adas.fit_resample(X_train_val, y_train_val)

            # Initialize XGBoost Classifier
            xgbc = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
            
            # Initialize RandomizedSearchCV
            rand_search = RandomizedSearchCV(estimator=xgbc, param_distributions=param_dist, cv=inner_kf, scoring='roc_auc', n_jobs=-1)
            rand_search.fit(X_train_res, y_train_res)
            best_params = rand_search.best_params_
            best_params_list.append(best_params)
            
            # Train the best model on the training set
            best_model = XGBClassifier(**best_params, use_label_encoder=False, eval_metric='logloss')
            best_model.fit(X_train_res, y_train_res)
            
            # Predict probabilities on the test set
            probs = best_model.predict_proba(X_test)[:, 1]
            y_pred_proba.extend(probs)

            # Threshold Adjustment (with cost-sensitive F1)
            precision, recall, thresholds = precision_recall_curve(y_test, probs)

            beta = 0.49  # Emphasize precision more than recall (adjust as needed)
            f1_scores = (1 + beta**2) * precision * recall / (beta**2 * precision + recall)
            best_threshold = thresholds[np.argmax(f1_scores)]

            # Predict using the best threshold
            predictions = (probs >= best_threshold).astype(int)
            y_true.extend(y_test)
            y_pred.extend(predictions)
    
    # Calculate metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print(f'Accuracy: {acc}')
    print(f'Precision: {prec}')
    print(f'Recall: {rec}')
    print(f'F1 Score: {f1}')
    print(f'AUC: {auc}')
    
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Creditworthy', 'Creditworthy'], yticklabels=['Non-Creditworthy', 'Creditworthy'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.show()

    # Plot Precision-Recall Curve
    plt.figure(figsize=(10, 7))
    plt.plot(recall, precision, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.show()

    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    plt.figure(figsize=(10, 7))
    plt.plot(fpr, tpr, marker='.')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.show()

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Non-Creditworthy', 'Creditworthy']))


    # Create a dictionary for the results
    results = {
        'acc': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': cm.tolist(),  # Convert numpy array to list for JSON serialization
    }

    # Save results to a JSON file
    with open('XGBoost_evaluation.json', 'w') as json_file:
        json.dump(results, json_file, indent=4)

    return best_model




