import pandas as pd
import numpy as np

#A1

 
def load_purchase_data(file_path="Lab-Session-Data.xlsx"):
    """Load the 'Purchase data' worksheet and return the feature and label matrices."""
    df = pd.read_excel(file_path, sheet_name='Purchase data')
    X = df.iloc[:, 1:4].values   # Quantities of Candies, Mangoes, Milk Packets
    y = df.iloc[:, 4].values.reshape(-1, 1)  # Payment column
    return X, y, df

def get_vector_space_properties(A):
    """Return (dimension, n_vectors, rank) of the matrix A."""
    dimension = A.shape[1]
    n_vectors = A.shape[0]
    rank = np.linalg.matrix_rank(A)
    return dimension, n_vectors, rank

def estimate_product_costs(A, C):
    """Estimate product costs using Moore-Penrose pseudo-inverse."""
    costs = np.linalg.pinv(A) @ C
    return costs.flatten()

#A2

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def add_rich_poor_labels(df):
    """Add 'Class' column to DataFrame: RICH if payment > 200; else POOR."""
    df = df.copy()
    df['Class'] = np.where(df.iloc[:, 4] > 200, 'RICH', 'POOR')
    return df

def train_classifier(A, labels):
    """Train and evaluate a logistic regression classifier; returns classification report text."""
    X_train, X_test, y_train, y_test = train_test_split(A, labels, test_size=0.2, random_state=42)
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred)
    return report

#A3

import statistics
import matplotlib.pyplot as plt
import seaborn as sns

def load_irctc_stock_data(file_path="Lab-Session-Data.xlsx"):
    """Load 'IRCTC Stock Price' worksheet as DataFrame."""
    return pd.read_excel(file_path, sheet_name='IRCTC Stock Price')

def get_price_mean_variance(df):
    """Return mean and variance of 'Price' column."""
    prices = df['Price']
    return statistics.mean(prices), statistics.variance(prices)

def wednesday_price_stats(df):
    """Return mean price for Wednesdays, and counts for Wednesday and all observations."""
    is_wed = df['Day'].str.startswith('Wed')
    wednesday_prices = df.loc[is_wed, 'Price']
    return statistics.mean(wednesday_prices), wednesday_prices.size, df.shape[0]

def april_price_mean(df):
    """Return mean price for April."""
    april_prices = df.loc[df['Month'] == 'Apr', 'Price']
    return statistics.mean(april_prices)

def loss_probability(df):
    """Return probability of loss (Chg% < 0) over the stock."""
    return (df['Chg%'] < 0).mean()

def profit_probability_wednesday(df):
    """Return probability of profit (Chg% > 0) on Wednesdays."""
    wed_mask = df['Day'].str.startswith('Wed')
    return (df.loc[wed_mask, 'Chg%'] > 0).mean()

def conditional_profit_given_wednesday(df):
    """Return P(Profit | Wednesday)."""
    wed_mask = df['Day'].str.startswith('Wed')
    return (df.loc[wed_mask, 'Chg%'] > 0).sum() / wed_mask.sum()

def plot_chg_vs_day(df):
    """Scatter plot: Chg% vs. Day of week."""
    plt.figure(figsize=(10, 5))
    sns.scatterplot(x=df['Day'], y=df['Chg%'])
    plt.title('Chg% vs. Day of Week')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

#A4

def load_thyroid_data(file_path="Lab-Session-Data.xlsx"):
    return pd.read_excel(file_path, sheet_name='thyroid0387_UCI')

def summarize_attributes(df):
    """Return summary table for each attribute: datatype, kind, encoding, missing, range."""
    summary = []
    for col in df.columns:
        col_data = df[col]
        dtype = col_data.dtype
        if dtype == object:
            uniq = set(col_data.dropna().unique())
            if uniq <= {'t', 'f'} or uniq <= {0, 1}:
                attr_type = 'binary'
                encoding = 'None'
            elif col_data.nunique() < 10:
                attr_type = 'nominal'
                encoding = 'One-hot'
            else:
                attr_type = 'categorical'
                encoding = 'Label'
            rng = None
        else:
            attr_type = 'numeric'
            encoding = 'None'
            rng = (col_data.min(), col_data.max())
        n_missing = col_data.isnull().sum() + np.sum(col_data == '?')
        summary.append({'column': col, 'dtype': str(dtype), 'attr_type': attr_type, 'encoding': encoding, 'missing': n_missing, 'range': rng})
    return pd.DataFrame(summary)

def numeric_stats(df):
    """Return mean and variance for numeric columns."""
    stats = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        stats[col] = {'mean': df[col].mean(), 'variance': df[col].var()}
    return stats

def outlier_count(df):
    """Return count of outliers (outside 1.5*IQR) for each numeric column."""
    outliers = {}
    for col in df.select_dtypes(include=[np.number]):
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        mask = (df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))
        outliers[col] = mask.sum()
    return outliers
