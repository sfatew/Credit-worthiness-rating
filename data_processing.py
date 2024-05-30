import pandas as pd
# Load Data 
data = pd.read_csv("data/german_credit.csv")
# Function to replace values in specified columns
def replace_values(df, col_replacements):
    for col, replacements in col_replacements.items():
        df[col] = df[col].replace(replacements)
    return df

# Dictionary specifying the columns and their replacement values
col_replacements = {
    'Account Balance': {4: 3},
    'Payment Status of Previous Credit': {0: 1, 4: 3},
    'Value Savings/Stocks': {4: 3, 5: 4},
    'Length of current employment': {2: 1, 3: 2, 4: 3, 5: 4},
    'Occupation': {2: 1, 3: 2, 4: 3},
    'Sex & Marital Status': {2: 1, 3: 2, 4: 3},
    'No of Credits at this Bank': {3: 2, 4: 2},
    'Guarantors': {3: 2},
    'Concurrent Credits': {2: 1, 3: 2},
    'Purpose': {i: 3 for i in range(4, 11)}
}

# Apply replacements
data = replace_values(data, col_replacements)

# Function to discretize continuous variables
def discretize_column(df, col, bins, labels):
    df[col] = pd.cut(df[col], bins=bins, labels=labels, right=False)
    return df

# Bins and labels for discretization
bins_labels = {
    'Duration of Credit (month)': {
        'bins': [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, float('inf')],
        'labels': [str(i) for i in range(10, 0, -1)]
    },
    'Credit Amount': {
        'bins': [0, 500, 1000, 1500, 2500, 5000, 7500, 10000, 15000, 20000, float('inf')],
        'labels': [str(i) for i in range(10, 0, -1)]
    },
    'Age (years)': {
        'bins': [0, 26, 40, 60, 65, float('inf')],
        'labels': [str(i) for i in range(1, 6)]
    }
}

# Apply discretization
for col, bl in bins_labels.items():
    data = discretize_column(data, col, bl['bins'], bl['labels'])

remove_cols = ['Foreign Worker', 'No of dependents', 'Guarantors','Duration in Current address']
new_data = data.copy()
new_data.drop(remove_cols,inplace= True, axis =1)
