import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import tree

# -----------------------
# Load dataset
# -----------------------
df = pd.read_csv("Project_labeled_features.csv")

# Target column (labelled by you)
target_col = "clarity_label"

# -----------------------
# A1. Entropy calculation
# -----------------------
def entropy(series):
    """Calculate entropy of a categorical pandas Series."""
    probs = series.value_counts(normalize=True)
    return -np.sum(probs * np.log2(probs))

# Equal-width binning
def equal_width_binning(series, bins=4):
    """Convert continuous data into categorical using equal-width binning."""
    return pd.cut(series, bins=bins, labels=False)

# Example: binning Clarity_Score if needed
df["Clarity_Score_binned"] = equal_width_binning(df["Clarity_Score"], bins=4)

# -----------------------
# A2. Gini Index
# -----------------------
def gini_index(series):
    """Calculate Gini index for a categorical pandas Series."""
    probs = series.value_counts(normalize=True)
    return 1 - np.sum(probs**2)

# -----------------------
# A3. Feature with max Information Gain
# -----------------------
def information_gain(df, feature, target):
    """Calculate information gain for a given feature."""
    total_entropy = entropy(df[target])
    values = df[feature].unique()
    weighted_entropy = 0
    for val in values:
        subset = df[df[feature] == val]
        weighted_entropy += (len(subset)/len(df)) * entropy(subset[target])
    return total_entropy - weighted_entropy

def find_best_feature(df, target):
    """Find feature with highest information gain."""
    gains = {}
    for col in df.columns:
        if col != target:
            if pd.api.types.is_numeric_dtype(df[col]):
                temp = equal_width_binning(df[col])
            else:
                temp = df[col]
            gains[col] = information_gain(df.assign(temp=temp), 'temp', target)
    return max(gains, key=gains.get), gains

best_feature, gains = find_best_feature(df, target_col)
print("Best root node feature:", best_feature)
print("Information Gains:", gains)

# -----------------------
# A4. Equal-width / Equal-frequency binning function
# -----------------------
def binning(series, bins=4, method='width'):
    """
    Bins a numeric series into categories.
    method: 'width' or 'frequency'
    """
    if method == 'width':
        return pd.cut(series, bins=bins, labels=False)
    elif method == 'frequency':
        return pd.qcut(series, q=bins, labels=False)
    else:
        raise ValueError("Method must be 'width' or 'frequency'")

# -----------------------
# A5. Build Decision Tree with sklearn
# -----------------------
# Encode categorical features if needed
df_encoded = df.copy()
for col in df_encoded.columns:
    if df_encoded[col].dtype == 'object':
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])

X = df_encoded.drop(columns=[target_col])
y = df_encoded[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = DecisionTreeClassifier(criterion='entropy', random_state=42)
clf.fit(X_train, y_train)

# -----------------------
# A6. Visualize Decision Tree
# -----------------------
plt.figure(figsize=(15,8))
plot_tree(clf, feature_names=X.columns, class_names=str(clf.classes_), filled=True)
plt.show()

# -----------------------
# A7. Decision Boundary for 2 features
# -----------------------
# Pick first two features for plotting
feat1, feat2 = X.columns[0], X.columns[1]
X2 = df_encoded[[feat1, feat2]]
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y, test_size=0.3, random_state=42)

clf2 = DecisionTreeClassifier(criterion='entropy', random_state=42)
clf2.fit(X_train2, y_train2)

# Meshgrid for decision boundary
x_min, x_max = X2[feat1].min() - 1, X2[feat1].max() + 1
y_min, y_max = X2[feat2].min() - 1, X2[feat2].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

Z = clf2.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(X2[feat1], X2[feat2], c=y, edgecolor='k', s=20)
plt.xlabel(feat1)
plt.ylabel(feat2)
plt.title("Decision Boundary (2 features)")
plt.show()
