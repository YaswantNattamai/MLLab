import pandas as pd
import numpy as np

# A1: Load Data, Vector Space Info, Cost Estimation
def load_data(file_path, sheet_name):
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    A = df.iloc[:, 1:4].to_numpy()  # Columns: Candies, Mangoes, Milk
    C = df.iloc[:, 4:5].to_numpy()  # Column: Payment
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

# A2: RICH/POOR Classification using if-else
def generate_rich_poor_labels(C):
    labels = [1 if payment > 200 else 0 for payment in C.flatten()]
    return labels

def classify_rich_poor_ifelse(A):
    predictions = []
    for row in A:
        total_items = sum(row)
        if total_items > 20:
            predictions.append(1)  # RICH
        else:
            predictions.append(0)  # POOR
    return predictions

def compute_accuracy(predicted, actual):
    correct = sum(p == a for p, a in zip(predicted, actual))
    return (correct / len(actual)) * 100

# === MAIN PROGRAM ===
file_path = "dataset.xlsx"
sheet_name = "Purchase data"

A, C, df = load_data(file_path, sheet_name)

# --- A1 Output ---
dim, n_vec, rank = get_vector_space_info(A)
X = estimate_product_costs(A, C)

print("===== A1: Linear Algebra Output =====")
print("Dimensionality of the Vector Space:", dim)
print("Vectors in the vector space:", n_vec)
print("Rank of matrix A:", rank)
print("Estimated product costs:", X)

# --- A2 Output ---
actual_labels = generate_rich_poor_labels(C)
predicted_labels = classify_rich_poor_ifelse(A)
accuracy = compute_accuracy(predicted_labels, actual_labels)

print("\n===== A2: RICH vs POOR Classification =====")
print("Actual Labels (RICH=1, POOR=0):", actual_labels)
print("Predicted Labels:", predicted_labels)
print(f"Accuracy of rule-based classifier: {accuracy:.2f}%")
