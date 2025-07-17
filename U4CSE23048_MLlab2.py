import pandas as pd
import numpy as np

def load_data(file_path, sheet_name):
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    A = df.iloc[:, 1:4].to_numpy()  # Columns: Candies, Mangoes, Milk
    C = df.iloc[:, 4:5].to_numpy()  # Column: Payment
    return A, C

def get_vector_space_info(A):
    dimensionality = len(A[0])
    num_vectors = len(A)
    rank = np.linalg.matrix_rank(A)
    return dimensionality, num_vectors, rank

def estimate_product_costs(A, C):
    pinverse = np.linalg.pinv(A)
    X = pinverse @ C
    return X.flatten()

# === MAIN PROGRAM ===
file_path = "Lab Session Data.xlsx"
sheet_name = "Purchase data"

A, C = load_data(file_path, sheet_name)
dim, n_vec, rank = get_vector_space_info(A)
X = estimate_product_costs(A, C)

print("Dimensionality of the Vector Space:", dim)
print("Vectors in the vector space:", n_vec)
print("Rank of matrix A:", rank)
print("Estimated product costs:", X)
