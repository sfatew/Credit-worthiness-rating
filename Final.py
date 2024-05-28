import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
import math
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE,ADASYN
import seaborn as sns

# Load Data 
data = pd.read_csv("data/german_credit.csv")
# Merge cell for discrete variables
data['Account Balance'] = data['Account Balance'].replace({4:3})
data['Payment Status of Previous Credit'] = data['Payment Status of Previous Credit'].replace({0:1,4:3})
data['Value Savings/Stocks'] = data['Value Savings/Stocks'].replace({4:3,5:4})
data['Length of current employment'] = data['Length of current employment'].replace({2:1,3:2,4:3,5:4})
data['Occupation'] = data['Occupation'].replace({2:1,3:2,4:3})
data['Sex & Marital Status'] = data['Sex & Marital Status'].replace({2:1,3:2,4:3})
data['No of Credits at this Bank'] = data['No of Credits at this Bank'].replace({2:1,3:1,4:1})
data['Guarantors'] = data['Guarantors'].replace({3:2})
data['Concurrent Credits'] = data['Concurrent Credits'].replace({2:1,3:2})
replacement_purpose = {i:3 for i in range(4, 11)}
data['Purpose'] = data['Purpose'].replace(replacement_purpose)
#Split continuous variables into discrete period
bin1 = [0,6,12,18,24,30,36,42,48,54,float('inf')]
label1 = [str(i) for i in range(10,0,-1)]
data['Duration of Credit (month)'] = pd.cut(data['Duration of Credit (month)'],bins = bin1, labels = label1,right = False)
bin2 = [0,500,1000,1500,2500,5000,7500,10000,15000,20000,float('inf')]
label2= label1
data['Credit Amount'] = pd.cut(data['Credit Amount'],bins = bin2, labels = label2, right =False)
bin3 = [0,26,40,60,65,float('inf')]
label3 = [str(i) for i in range(1,6)]
data['Age (years)'] = pd.cut(data['Age (years)'],bins = bin3, labels = label3, right = False)

x = data.iloc[:,1:].values
y = data.iloc[:,0].values

def data_info(data):
    data.info()
    print(data.shape)
    print(data.describe())
    print(data["Creditability"].value_counts())
    #Visualising the label column
    plt.pie(data["Creditability"].value_counts(),labels = ["High Credit", "Low Credit"],colors =["green","red"], autopct = '%1.1f%%',radius = 1)
    plt.legend(title= "Credit Scoring")
    #Correlation Matrix 
    correlation_matrix = data.corr()
    fig = plt.figure(figsize=(13,8))
    heatmap=sns.heatmap(correlation_matrix,annot = True,square= True, linewidths=.5,cmap=plt.cm.Reds,annot_kws={"size":6}) 
    heatmap.set_aspect('equal', 'box')  
    plt.show()

#Balancing the dataset

ada = ADASYN(random_state=15)
x,y = ada.fit_resample(x,y)

#Standard Scaler 
scaler = StandardScaler()
x = scaler.fit_transform(x)
#Training 
#Create Stratified K-fold Object
skf = RepeatedStratifiedKFold(n_splits = 10 , random_state=1)

conf_matrices = []
precision_list = []
recall_list = []
f1_list = []
tprs = []
aucs = []
mean_fprs = np.linspace(0,1,100)
true_positive_prob = []
false_positive_prob = []

for train_index, test_index in tqdm(skf.split(x,y)):
    x_train, x_test = x[train_index],x[test_index]
    y_train, y_test = y[train_index],y[test_index]
    rf = RandomForestClassifier(n_estimators=400,criterion="gini",max_features=10, min_samples_leaf=1,min_samples_split=10,n_jobs=-1)
    rf.fit(x_train,y_train)
    y_pred = rf.predict(x_test)
    
    # Probabilities for ROC Curve
    y_pred_proba = rf.predict_proba(x_test)[:,1]

    #Confusion Matrix 
    conf_matrix = confusion_matrix(y_test,y_pred)
    conf_matrices.append(conf_matrix)

    #Classification reports
    report = classification_report(y_test, y_pred, output_dict=True)
    precision_list.append(report['weighted avg']['precision'])
    recall_list.append(report['weighted avg']['recall'])
    f1_list.append(report['weighted avg']['f1-score'])

    # Compute ROC curve and area the curve
    fpr,tpr,thres = roc_curve(y_test,y_pred_proba)
    aucs.append(auc(fpr,tpr))
    tprs.append(np.interp(mean_fprs,fpr,tpr))
    tprs[-1][0] = 0.0

    # Collect probabilities for true positive prob and false positive prob
    true_positive_prob.extend(y_pred_proba[y_test == 1])
    false_positive_prob.extend(y_pred_proba[y_test == 0])

#Mean of confusion Matrix
mean_conf_matrix = np.mean(conf_matrices,axis =0)
labels = ["Good","Bad"]
plt.figure(figsize=(6,4))
sns.heatmap(mean_conf_matrix, xticklabels=labels, yticklabels=labels,annot = True, fmt = 'g')
plt.title("Confusion Matrix")
plt.ylabel("True Class")
plt.xlabel("Predicted Class")
plt.tight_layout

#Run classification metrics 
print("Average Precision: {:.4f}".format(np.mean(precision_list)))
print("Average Recall: {:.4f}".format(np.mean(recall_list)))
print("Average F1-score: {:.4f}".format(np.mean(f1_list)))

#Plot AUROC
plt.figure(figsize = (15,8))
plt.plot([0,1],[0,1],linestyle ="--",color= 'red',lw = 2,label = 'Chance',alpha = .8)
mean_tprs = np.mean(tprs, axis = 0)
mean_auc = auc(mean_fprs, mean_tprs)
std_auc = np.std(aucs)
std_tpr = np.std(tprs,axis = 0)
tprs_upper = np.minimum(mean_tprs + std_tpr, 1)
tprs_lower = np.maximum(mean_tprs - std_tpr,0)
plt.fill_between(mean_fprs,tprs_upper,tprs_lower,color = 'grey',alpha =.2, label = r'$\pm$ 1 std.dev.')

mean_tprs[-1] = 1
plt.plot(mean_fprs, mean_tprs, lw = 2, alpha =.8,color = 'navy',label =r'mean ROC (AUC ={:.2f} $\pm$ {:.2f})'.format(mean_auc,std_auc))
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc = "lower right" )
plt.show()

#Plot PDF with Decision Threshold
plt.figure(figsize = (15,8))
sns.kdeplot(true_positive_prob, label = 'True Positive', color = 'b')
sns.kdeplot(false_positive_prob, label = 'False Positive', color = 'r')
decision_threshold = 0.5
plt.axvline(x = decision_threshold, color = 'g', label = f'Decision Threshold {decision_threshold}') 
plt.xlabel("Predicted Probability")
plt.ylabel("PDF")
plt.title("PDF with Decision Threshold")
plt.legend(loc = "upper right")
plt.show()
