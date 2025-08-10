import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler

# ----------------------
# Load dataset globally
# ----------------------
df = pd.read_csv("Project_labeled_features.csv")

# -----------------------------------
# A1: Linear Regression, 1 feature
# -----------------------------------
def a1_reg_single_feature():
    X = df[["mfcc_1"]]
    y = df["clarity_score"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LinearRegression().fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test

# -----------------------------------
# A2: Regression metrics, single feature
# -----------------------------------
def a2_print_single_metrics(model, X_train, X_test, y_train, y_test):
    print("A2: Regression using mfcc_1 only")
    for setname, X_set, y_set in [
        ("Train", X_train, y_train), ("Test", X_test, y_test)
    ]:
        preds = model.predict(X_set)
        print(f"  {setname}:")
        print(f"    MSE: {mean_squared_error(y_set, preds):.2f}")
        print(f"    RMSE: {np.sqrt(mean_squared_error(y_set, preds)):.2f}")
        print(f"    MAPE: {mean_absolute_percentage_error(y_set, preds):.4f}")
        print(f"    R2: {r2_score(y_set, preds):.4f}")

# -----------------------------------
# A3: Linear Regression, all features
# -----------------------------------
def a3_reg_all_features():
    features = [col for col in df.columns if col not in ["id", "clarity_score", "clarity_label"]]
    X = df[features]
    y = df["clarity_score"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LinearRegression().fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test

def a3_print_all_metrics(model, X_train, X_test, y_train, y_test):
    print("\nA3: Regression using all numerical features")
    for setname, X_set, y_set in [
        ("Train", X_train, y_train), ("Test", X_test, y_test)
    ]:
        preds = model.predict(X_set)
        print(f"  {setname}:")
        print(f"    MSE: {mean_squared_error(y_set, preds):.2f}")
        print(f"    RMSE: {np.sqrt(mean_squared_error(y_set, preds)):.2f}")
        print(f"    MAPE: {mean_absolute_percentage_error(y_set, preds):.4f}")
        print(f"    R2: {r2_score(y_set, preds):.4f}")

# -----------------------------------
# Helper: get scaled clustering features
# -----------------------------------
def get_scaled_clustering_features():
    features = [col for col in df.columns if col not in ["id", "clarity_score", "clarity_label"]]
    X_cluster = df[features]
    scaler = StandardScaler()
    X_cluster_scaled = scaler.fit_transform(X_cluster)
    return X_cluster_scaled

# -----------------------------------
# A4: KMeans clustering (k=2)
# -----------------------------------
def a4_kmeans_k2():
    X_cluster = get_scaled_clustering_features()
    kmeans = KMeans(n_clusters=2, random_state=42, n_init='auto').fit(X_cluster)
    print("\nA4: First 10 cluster labels (k=2):", kmeans.labels_[:10])
    return kmeans, X_cluster

# -----------------------------------
# A5: Clustering metrics for k=2
# -----------------------------------
def a5_clustering_metrics(kmeans, X_cluster):
    print("\nA5: Clustering metrics for k=2")
    print(f"  Silhouette: {silhouette_score(X_cluster, kmeans.labels_):.4f}")
    print(f"  Calinski-Harabasz: {calinski_harabasz_score(X_cluster, kmeans.labels_):.2f}")
    print(f"  Davies-Bouldin: {davies_bouldin_score(X_cluster, kmeans.labels_):.4f}")

# -----------------------------------
# A6: Clustering metrics for k=2..5
# -----------------------------------
def a6_kmeans_metrics():
    X_cluster = get_scaled_clustering_features()
    print("\nA6: Clustering metrics for k=2 to k=5")
    for k in range(2, 6):
        km = KMeans(n_clusters=k, random_state=42, n_init='auto').fit(X_cluster)
        print(f"  k={k}: Silhouette={silhouette_score(X_cluster, km.labels_):.4f}, "
              f"CH={calinski_harabasz_score(X_cluster, km.labels_):.2f}, "
              f"DB={davies_bouldin_score(X_cluster, km.labels_):.4f}")

# -----------------------------------
# A7: Elbow plot for KMeans, for k in range(2, 21)
# -----------------------------------
def a7_elbow_plot():
    X_cluster = get_scaled_clustering_features()
    distortions = []
    for k in range(2, 21):
        km = KMeans(n_clusters=k, random_state=42, n_init='auto').fit(X_cluster)
        distortions.append(km.inertia_)
    plt.plot(range(2, 21), distortions, marker='o')
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Distortion (Inertia)")
    plt.title("Elbow Plot for k-means")
    plt.grid(True)
    plt.show()

# -----------------------------------
# Main routine: executes everything in order
# -----------------------------------
if __name__ == "__main__":
    # A1 & A2
    model1, X_train1, X_test1, y_train1, y_test1 = a1_reg_single_feature()
    a2_print_single_metrics(model1, X_train1, X_test1, y_train1, y_test1)
    # A3
    model2, X_train2, X_test2, y_train2, y_test2 = a3_reg_all_features()
    a3_print_all_metrics(model2, X_train2, X_test2, y_train2, y_test2)
    # A4, A5
    kmeans2, X_cluster = a4_kmeans_k2()
    a5_clustering_metrics(kmeans2, X_cluster)
    # A6
    a6_kmeans_metrics()
    # A7
    print("\nA7: Elbow Plot (k=2 to k=20)")
    a7_elbow_plot()
