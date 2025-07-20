import pandas as pd

# loading  data and marking '?' as NaN
file_path = "Lab-Session-Data.xlsx"
df = pd.read_excel(file_path, sheet_name="thyroid0387_UCI", na_values=['?'])


# data types of every column
print("\nData Types:")
print(df.dtypes)

# checking for missing values
print("\nMissing Values in Each Column:")
print(df.isnull().sum())

# Step 5: Mean and variance for numeric columns
print("\nMean and Variance of Numeric Columns:")
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
for col in numeric_cols:
    mean = df[col].mean()
    var = df[col].var()
    print(f"{col}: Mean = {mean:.2f}, Variance = {var:.2f}")

# range of numeric data
print("\nOutlier Check (Min/Max values):")
for col in numeric_cols:
    print(f"{col}: Min = {df[col].min()}, Max = {df[col].max()}")

# suggested encoding for categorical columns
print("\nSuggested Encoding for Categorical Columns:")
for col in df.select_dtypes(include='object').columns:
    unique_vals = df[col].nunique()
    if unique_vals <= 5:
        print(f"{col}: Label Encoding")
    else:
        print(f"{col}: One-Hot Encoding")