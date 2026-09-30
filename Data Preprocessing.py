# Data Processing Code For Machine Learning

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    LabelEncoder, OneHotEncoder, OrdinalEncoder
)
from sklearn.impute import SimpleImputer

# Sample data
data = {
    'Age': [25, 30, np.nan, 35, 28, 40],
    'Salary': [50000, 60000, 55000, np.nan, 65000, 70000],
    'Gender': ['Male', 'Female', 'Female', 'Male', 'Female', 'Male'],
    'City': ['Nawabshah', 'Lahore', 'Karachi', 'Hyderabad', 'Dadu', 'Sukkur'],
    'Purchased': ['No', 'Yes', 'No', 'Yes', 'Yes', 'No']
}
df = pd.DataFrame(data)

# 1. Train-Test Split
X = df.drop('Purchased', axis=1)
y = df['Purchased']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. Handling Missing Values
# Numeric: mean/median imputation
num_imputer = SimpleImputer(strategy='median')
X_train[['Age', 'Salary']] = num_imputer.fit_transform(X_train[['Age', 'Salary']])
X_test[['Age', 'Salary']] = num_imputer.transform(X_test[['Age', 'Salary']])

# Categorical: most frequent
cat_imputer = SimpleImputer(strategy='most_frequent')
X_train[['Gender', 'City']] = cat_imputer.fit_transform(X_train[['Gender', 'City']])

# 3. Encoding Categorical Variables
# Label Encoding (target variable)
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

# One-Hot Encoding (nominal features)
X_train_ohe = pd.get_dummies(X_train, columns=['Gender', 'City'], drop_first=True)
X_test_ohe = pd.get_dummies(X_test, columns=['Gender', 'City'], drop_first=True)
# Align columns
X_test_ohe = X_test_ohe.reindex(columns=X_train_ohe.columns, fill_value=0)

# 4. Feature Scaling
scaler = StandardScaler()   # or MinMaxScaler(), RobustScaler()
X_train_scaled = scaler.fit_transform(X_train_ohe)
X_test_scaled = scaler.transform(X_test_ohe)

print("Train shape:", X_train_scaled.shape)
print("Test shape:", X_test_scaled.shape)


