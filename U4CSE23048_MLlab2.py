import pandas as pd
import numpy as np

# === A1: Linear Algebra ===
def load_data(file_path, sheet_name):
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    A = df.iloc[:, 1:4].to_numpy()  # Features: Candies, Mangoes, Milk
    C = df.iloc[:, 4:5].to_numpy()  # Payment
    return A, C, df

def get_vector_space_info(A):
    dimensionality = len(A[0])
    num_vectors = len(A)
    rank = np.linalg.matrix_rank(A)
    return dimensionality, num_vectors, rank

def estimate_product_costs(A, C):
    pinverse = np.linalg.pinv(A)
    X = pinverse @ C
    return X.flatten()

# === A2: Exact Labeling ===
def generate_rich_poor_labels(C):
    return [1 if payment > 200 else 0 for payment in C.flatten()]

# === MAIN PROGRAM ===
file_path = "Lab Session Data.xlsx"
sheet_name = "Purchase data"

A, C, df = load_data(file_path, sheet_name)

# A1 Output
dim, n_vec, rank = get_vector_space_info(A)
X = estimate_product_costs(A, C)

# A2 Output (Exact RICH/POOR labels)
labels = generate_rich_poor_labels(C)

# Final Output
print("===== A1: Linear Algebra Output =====")
print("Dimensionality of the Vector Space:", dim)
print("Vectors in the vector space:", n_vec)
print("Rank of matrix A:", rank)
print("Estimated product costs:", X)

print("\n===== A2: Customer Labels (RICH=1, POOR=0) =====")
print("Customer Labels:", labels)
