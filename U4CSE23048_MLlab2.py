import pandas as pd
import numpy as np

def load_purchase_data(filepath):
    """Loads the 'Purchase data' sheet and returns matrix A and target vector C."""
    df = pd.read_excel(filepath, sheet_name="Purchase data")
    A = df.iloc[:, 1:-1].values  # Exclude first column (ID) and last column (Total)
    C = df.iloc[:, -1].values    # Total column
    return A, C, df

def clean_input_matrices(A, C):
    """Converts all values to float and replaces NaN/Inf/-Inf with 0."""
    A_clean = np.nan_to_num(A.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    C_clean = np.nan_to_num(C.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    return A_clean, C_clean

def get_vector_space_details(A):
    """Returns dimension, number of vectors, and rank of matrix A."""
    dimension = A.shape[1]           # Number of columns
    num_vectors = A.shape[0]         # Number of rows
    rank = np.linalg.matrix_rank(A)  # Rank
    return dimension, num_vectors, rank

def calculate_cost_using_pinv(A, C):
    """Computes the pseudo-inverse solution X for AX = C."""
    pseudo_inv = np.linalg.pinv(A)
    X = np.dot(pseudo_inv, C)
    return X

# ========== MAIN EXECUTION ==========
file_path = "Lab Session Data.xlsx"  # Must be in the same folder

# Load and clean data
A, C, df_purchase = load_purchase_data(file_path)
A, C = clean_input_matrices(A, C)

# Debugging checks
print("\nSample of cleaned A matrix (first 2 rows):")
print(A[:2])
print("\nSample of cleaned C vector (first 2 values):")
print(C[:2])

# Get vector space properties
dim, n_vec, rank = get_vector_space_details(A)

# Compute cost vector
X = calculate_cost_using_pinv(A, C)

# Final output
print("\n=== Linear Algebra Summary ===")
print("Vector space dimension:", dim)
print("Number of vectors:", n_vec)
print("Rank of matrix A:", rank)
print("\nEstimated cost of each product (X):")
print(X)
