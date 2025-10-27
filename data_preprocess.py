# Import all packages/library.
import pandas as pd

# Read the raw data file.
raw_df = pd.read_csv('./data/retail_sales_synthetic.csv')
df = raw_df.copy()  # Copy to ensure every change made in this code doesn't affect the raw data.
# Check the raw data information before preprocessing.
print('\nInitial Dataframe Information:')
print(df.info())
# It will be truncated if we printed it as df.head() or df.describe() so we can't check (see) all columns.
# In order to avoid that, we need to print it partially.
n_col = len(df.columns)
print('\nInitial Dataframe Head:')
print(df.head().iloc[:, :int(n_col/2)])
print(df.head().iloc[:, int(n_col/2):])
print('\nInitial Dataframe Numeric Stats:')
print(df.describe().iloc[:, :int(n_col/3)])
print(df.describe().iloc[:, int(n_col/3):])

# test = df['date'].nunique()
# print(test)