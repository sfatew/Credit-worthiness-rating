import pandas as pd

# Load Data 
data = pd.read_csv("data\\german_credit.csv")
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
    'No of Credits at this Bank': {2: 1, 3: 1, 4: 1},
    'Guarantors': {3: 2},
    'Concurrent Credits': {2: 1, 3: 2},
    'Purpose': {i: 3 for i in range(4, 11)}
}

# Apply replacements
data = replace_values(data, col_replacements)


remove_cols = ['Foreign Worker', 'No of dependents', 'Guarantors','Duration in Current address']
new_data = data.copy()
new_data.drop(remove_cols,inplace= True, axis =1)

new_data.to_csv("data\\new_german_credit.csv")