# Import necessary libraries
import pandas as pd
import numpy as np

# Load the earnings.csv file
# Using sep=';' because the file is semicolon-delimited
df = pd.read_csv('earnings.csv', sep=';')

# Display basic information about the dataframe
print("Basic information about the dataframe:")
print(f"Shape of the dataframe: {df.shape}")
print(f"Number of observations (rows): {len(df)}")
print(f"Number of variables (columns): {len(df.columns)}")
print("\nColumn names:")
print(df.columns.tolist())

# Check for missing values
print("\nChecking for missing values:")
missing_values = df.isnull().sum()
missing_percentage = (df.isnull().sum() / len(df)) * 100

# Create a summary of missing values
missing_summary = pd.DataFrame({
    'Column': df.columns,
    'Missing Values': missing_values,
    'Percentage Missing': missing_percentage.round(2)
})

print(missing_summary)

# Check if there are any missing values in the entire dataset
total_missing = df.isnull().sum().sum()
print(f"\nTotal missing values in the dataset: {total_missing}")
if total_missing == 0:
    print("The dataset has no missing values.")
else:
    print(f"The dataset has {total_missing} missing values across all columns.")

# Display the first few rows of the dataframe to see the data
print("\nFirst 5 rows of the dataframe:")
print(df.head())

# Display summary statistics
print("\nSummary statistics for numeric columns:")
print(df.describe())