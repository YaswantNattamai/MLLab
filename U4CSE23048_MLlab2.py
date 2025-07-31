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
from sklearn.preprocessing import LabelEncoder

def load_irctc_stock_data(file_path="Lab-Session-Data.xlsx"):
    """Load 'IRCTC Stock Price' worksheet as DataFrame."""
    return pd.read_excel(file_path, sheet_name='IRCTC Stock Price')

def get_price_mean_variance(df):
    """Return mean and variance of 'Price' column."""
    prices = df['Price']
    return statistics.mean(prices), statistics.variance(prices)

def wednesday_price_stats(df):
    is_wed = df['Day'].str.startswith('Wed')
    wednesday_prices = df.loc[is_wed, 'Price']
    return statistics.mean(wednesday_prices), wednesday_prices.size, df.shape[0]

def april_price_mean(df):
    april_prices = df.loc[df['Month'] == 'Apr', 'Price']
    return statistics.mean(april_prices)

def loss_probability(df):
    return (df['Chg%'] < 0).mean()

def profit_probability_wednesday(df):
    wed_mask = df['Day'].str.startswith('Wed')
    return (df.loc[wed_mask, 'Chg%'] > 0).mean()

def conditional_profit_given_wednesday(df):
    wed_mask = df['Day'].str.startswith('Wed')
    return (df.loc[wed_mask, 'Chg%'] > 0).sum() / wed_mask.sum()

def plot_chg_vs_day(df):
    plt.figure(figsize=(10, 5))
    sns.scatterplot(x=df['Day'], y=df['Chg%'])
    plt.title('Chg% vs. Day of Week')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

#A4

def load_thyroid_data(file_path="Lab-Session-Data.xlsx"):
    return pd.read_excel(file_path, sheet_name='thyroid0387_UCI',na_values=['?'])
    
def summarize_attributes(df):
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
    stats = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        stats[col] = {'mean': df[col].mean(), 'variance': df[col].var()}
    return stats

def outlier_count(df):
    outliers = {}
    for col in df.select_dtypes(include=[np.number]):
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        mask = (df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))
        outliers[col] = mask.sum()
    return outliers


#A5

def get_first_two_binary_vectors(df):
    binary_cols = [
        col for col in df.columns
        if set(df[col].dropna().unique()) <= {'t', 'f', 0, 1}
    ]
    binmat = df[binary_cols].replace({'t': 1, 'f': 0}).astype(int)
    return binmat.iloc[0].values, binmat.iloc[1].values

def jaccard_coefficient(vec1, vec2):
    f11 = np.sum((vec1 == 1) & (vec2 == 1))
    f10 = np.sum((vec1 == 1) & (vec2 == 0))
    f01 = np.sum((vec1 == 0) & (vec2 == 1))
    denom = f11 + f10 + f01
    return f11 / denom if denom != 0 else np.nan

def smc_coefficient(vec1, vec2):
    f11 = np.sum((vec1 == 1) & (vec2 == 1))
    f00 = np.sum((vec1 == 0) & (vec2 == 0))
    f10 = np.sum((vec1 == 1) & (vec2 == 0))
    f01 = np.sum((vec1 == 0) & (vec2 == 1))
    total = f11 + f00 + f10 + f01
    return (f11 + f00) / total if total != 0 else np.nan

'''                                                     A6                                                                            '''
import pandas as pd
import numpy as np

# Load the dataset (adjust the filename as needed)
df_thyroid = pd.read_excel('dataset.xlsx', sheet_name='thyroid0387_UCI')


def cosineSimilarity(df):
    # Select two observations (replace 0 and 1 with appropriate indices of the desired records)
    vec1 = df.iloc[0]
    vec2 = df.iloc[1]

    # Select only numerical columns for the vectors (drop non-numeric columns like 'sex', 'Condition', etc.)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    vector1 = vec1[numeric_cols].values.astype(float)
    vector2 = vec2[numeric_cols].values.astype(float)

    # Calculate Cosine Similarity
    dot_product = np.dot(vector1, vector2)
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)
    cosine_similarity = dot_product / (norm1 * norm2)

    print("Cosine Similarity", cosine_similarity)


# A7
def analyze_similarity_metrics(file_path, sheet_name, save_path=None):
    # Load dataset
    df = pd.read_excel(file_path, sheet_name=sheet_name, na_values=["?"])

    # Handle missing values and encode categorical columns
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0])
            df[col] = LabelEncoder().fit_transform(df[col])
        else:
            df[col] = df[col].fillna(df[col].mean())

    # Select first 20 observations
    df_20 = df.iloc[:20].reset_index(drop=True)

    # Cosine Similarity
    cos_sim_matrix = cosineSimilarity(df_20)

    # Identify binary columns
    binary_cols = [col for col in df_20.columns if set(df_20[col].unique()).issubset({0, 1})]

    # Define Jaccard and SMC
    def jaccard(a, b):
        f11 = np.sum((a == 1) & (b == 1))
        f10 = np.sum((a == 1) & (b == 0))
        f01 = np.sum((a == 0) & (b == 1))
        denom = f11 + f10 + f01
        return f11 / denom if denom != 0 else 0

    def smc(a, b):
        f11 = np.sum((a == 1) & (b == 1))
        f00 = np.sum((a == 0) & (b == 0))
        f10 = np.sum((a == 1) & (b == 0))
        f01 = np.sum((a == 0) & (b == 1))
        total = f11 + f00 + f10 + f01
        return (f11 + f00) / total if total != 0 else 0

    # Compute JC and SMC matrices
    n = len(df_20)
    jc_matrix = np.zeros((n, n))
    smc_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            vec1 = df_20.loc[i, binary_cols].values
            vec2 = df_20.loc[j, binary_cols].values
            jc_matrix[i, j] = jaccard(vec1, vec2)
            smc_matrix[i, j] = smc(vec1, vec2)

    # Plot heatmaps
    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    sns.heatmap(jc_matrix, annot=False, cmap='Blues', square=True, cbar=True)
    plt.title("Jaccard Coefficient")

    plt.subplot(1, 3, 2)
    sns.heatmap(smc_matrix, annot=False, cmap='Greens', square=True, cbar=True)
    plt.title("Simple Matching Coefficient")

    plt.subplot(1, 3, 3)
    sns.heatmap(cos_sim_matrix, annot=False, cmap='Reds', square=True, cbar=True)
    plt.title("Cosine Similarity")

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

#A8
import pandas as pd
import numpy as np

def impute_missing_values(file_path, sheet_name, na_values=["?"], return_df=False):
    # Load the data
    df = pd.read_excel(file_path, sheet_name=sheet_name, na_values=na_values)
    filled_columns = []

    # Impute categorical columns with Mode
    cat_cols = df.select_dtypes(include='object').columns
    df[cat_cols] = df[cat_cols].apply(lambda col: col.fillna(col.mode()[0]))
    filled_columns += [f"{col} (Mode)" for col in cat_cols if df[col].isnull().sum() == 0]

    # Helper function to impute numeric columns based on outlier presence
    def impute_numeric(col):
        if col.isnull().sum() == 0:
            return col
        q1, q3 = col.quantile([0.25, 0.75])
        iqr = q3 - q1
        has_outlier = ((col < (q1 - 1.5 * iqr)) | (col > (q3 + 1.5 * iqr))).any()
        method = "Median" if has_outlier else "Mean"
        filled_columns.append(f"{col.name} ({method})")
        return col.fillna(col.median() if has_outlier else col.mean())

    # Apply to numeric columns
    num_cols = df.select_dtypes(include=np.number).columns
    df[num_cols] = df[num_cols].apply(impute_numeric)

    # Print summary
    print("Imputation complete.\nFilled columns and methods used:")
    for col in filled_columns:
        print(f"→ {col}")

    remaining_missing = df.isnull().sum().sum()
    if remaining_missing == 0:
        print("\nAll missing values successfully imputed.")
    else:
        print(f"\nThere are still {remaining_missing} missing values remaining.")

    return df if return_df else None

#A9
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def impute_and_normalize(file_path, sheet_name, na_values=["?"], return_df=False):
    # Load dataset
    df = pd.read_excel(file_path, sheet_name=sheet_name, na_values=na_values)

    # Impute categorical columns with mode
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Impute numerical columns with mean or median based on outliers
    for col in df.select_dtypes(include=np.number).columns:
        if df[col].isnull().sum() == 0:
            continue
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        has_outlier = ((df[col] < (q1 - 1.5 * iqr)) | (df[col] > (q3 + 1.5 * iqr))).any()
        df[col] = df[col].fillna(df[col].median() if has_outlier else df[col].mean())

    # Normalize numeric columns using Min-Max scaling
    numeric_cols = df.select_dtypes(include=np.number).columns
    scaler = MinMaxScaler()
    df_normalized = df.copy()
    df_normalized[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    print("Normalization complete. Sample normalized data:")
    print(df_normalized[numeric_cols].head())

    return df_normalized if return_df else None



# --------- A1: Purchase Data Analysis ---------
A, C, purchase_df = load_purchase_data()
dim, n_vecs, rk = get_vector_space_properties(A)
costs = estimate_product_costs(A, C)
print("A1: PURCHASE DATA ANALYSIS")
print(f"Dimensionality of vector space: {dim}")
print(f"Number of vectors: {n_vecs}")
print(f"Rank of Purchase Quantity Matrix: {rk}")
print(f"Estimated product costs: {costs}\n")

# --------- A2: Rich/Poor Classifier ---------
purchase_df = add_rich_poor_labels(purchase_df)
report = train_classifier(A, purchase_df['Class'])
print("A2: RICH/POOR CLASSIFICATION REPORT:")
print(report)

# --------- A3: IRCTC Stock Data Analysis ---------
stock_df = load_irctc_stock_data()
mean_price, var_price = get_price_mean_variance(stock_df)
wed_mean, n_wed, n_all = wednesday_price_stats(stock_df)
april_mean = april_price_mean(stock_df)
loss_prob = loss_probability(stock_df)
profit_wed_prob = profit_probability_wednesday(stock_df)
cond_prob = conditional_profit_given_wednesday(stock_df)
print("A3: IRCTC STOCK DATA ANALYSIS")
print(f"Mean price: {mean_price}, Variance: {var_price}")
print(f"Wednesday mean price: {wed_mean} (Wednesdays: {n_wed}, Total: {n_all})")
print(f"April mean price: {april_mean}")
print(f"Probability of Loss: {loss_prob:.2f}")
print(f"Probability of Profit on Wednesday: {profit_wed_prob:.2f}")
print(f"Conditional P(Profit|Wednesday): {cond_prob:.2f}")

print("Plotting Chg% vs Day of Week...")
plot_chg_vs_day(stock_df)  # This will show a plot

# --------- A4: Thyroid Data Exploration ---------
thy_df = load_thyroid_data()
summary_df = summarize_attributes(thy_df)
num_stats = numeric_stats(thy_df)
outliers = outlier_count(thy_df)
print("A4: THYROID DATA SUMMARY")
print("Attribute summary:\n", summary_df)
print("Numeric mean and variance:\n", num_stats)
print("Outlier count (numeric cols):\n", outliers)

# --------- A5: Jaccard & SMC Similarity ---------
vec1, vec2 = get_first_two_binary_vectors(thy_df)
jc = jaccard_coefficient(vec1, vec2)
smc = smc_coefficient(vec1, vec2)
print("A5: SIMILARITY MEASURES")
print(f"Jaccard coefficient: {jc:.3f}")
print(f"Simple Matching coefficient: {smc:.3f}")
if jc < smc:
    print("Jaccard is stricter: only positive matches; SMC includes negatives.")
else:
    print("Both coefficients indicate similarity, but are used differently.")


