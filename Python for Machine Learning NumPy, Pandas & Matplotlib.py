# NumPy, Pandas & Matplotlib Code For Machine Learning


# NumPy — Numerical Computing
import numpy as np

arr = np.array([1, 2, 3, 4, 5])
print(arr.shape, arr.dtype)

# 2D arrays
matrix = np.array([[1, 2, 3], [4, 5, 6]])
print(matrix.shape)  # (2, 3)

# Useful operations
print(np.zeros((3, 3)))
print(np.ones((2, 4)))
print(np.eye(3))                    # Identity matrix
print(np.arange(0, 10, 2))          # [0 2 4 6 8]
print(np.linspace(0, 1, 5))         # 5 evenly spaced numbers

# Broadcasting & vectorization
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(a + b, a * b, a @ b)          # Element-wise, dot product
print(np.dot(a, b))
print(a.reshape(-1, 1))             # Column vector

# Random
np.random.seed(42)
rand = np.random.randn(3, 3)        # Standard normal
print(rand.mean(), rand.std())

# Indexing & slicing
print(matrix[0, :])                 # First row
print(matrix[:, 1])                 # Second column
print(matrix[matrix > 3])           # Boolean indexing

# Pandas — Data Manipulation
import pandas as pd

df = pd.DataFrame({
    'Name': ['Samay', 'Zakir', 'Tanmay', 'Kullu'],
    'Age': [29, 31, 37, 28],
    'Salary': [500000, 600000, 7500000, 650000],
    'Occupation': ['Comedian', 'Comedian', 'YouTuber', 'Comedian']
})

print(df.head())
print(df.info())
print(df.describe())

# Selection
print(df['Age'])                    # Series
print(df[['Name', 'Salary']])       # DataFrame
print(df.loc[0])                    # Row by label
print(df.iloc[1:3])                 # Row by position
print(df[df['Age'] > 28])           # Filtering

# Operations
df['Bonus'] = df['Salary'] * 0.10
df['Senior'] = df['Age'] > 30
print(df.groupby('Occupation')['Salary'].mean())
print(df.sort_values('Salary', ascending=False))

# Handling missing values
df.loc[1, 'Salary'] = np.nan
print(df.isnull().sum())
df['Salary'].fillna(df['Salary'].mean(), inplace=True)

# Matplotlib / Seaborn — Visualization
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot([1, 2, 3, 4], [1, 4, 9, 16], 'b-o', label='y=x²')
plt.xlabel('X'); plt.ylabel('Y'); plt.legend(); plt.title('Line Plot')

plt.subplot(1, 2, 2)
plt.scatter(df['Age'], df['Salary'], c='red', s=100)
plt.xlabel('Age'); plt.ylabel('Salary'); plt.title('Scatter')

plt.tight_layout()
plt.show()

# Seaborn
sns.set_style('whitegrid')
sns.histplot(df['Salary'], kde=True)
sns.boxplot(x='Occupation', y='Salary', data=df)
sns.heatmap(df[['Age', 'Salary', 'Bonus']].corr(), annot=True)
plt.show()
