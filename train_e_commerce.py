from matplotlib import pyplot as plt
import numpy as np
from LogisticRegression import LogisticRegression

data = np.genfromtxt("customers.csv", delimiter=",", dtype=str, skip_header=1)

# ---- Labels ----
y_raw = data[:, -1]

# Satisfied = 1, else = 0
y = (y_raw == "Satisfied").astype(float)

# Features 
X_raw = data[:, :-1]  # everything except satisfaction

gender = X_raw[:, 1]
age = X_raw[:, 2].astype(float)
city = X_raw[:, 3]
membership = X_raw[:, 4]
total_spend = X_raw[:, 5].astype(float)
items = X_raw[:, 6].astype(float)
rating = X_raw[:, 7].astype(float)
discount = X_raw[:, 8]
days = X_raw[:, 9].astype(float)

# Binary encodings
gender_enc = np.where(gender == "Male", 1, 0).reshape(-1, 1)
discount_enc = np.where(discount == "TRUE", 1, 0).reshape(-1, 1)

# One-hot city
cities = np.unique(city)
city_enc = np.zeros((len(city), len(cities)))
for i, c in enumerate(cities):
    city_enc[:, i] = (city == c).astype(float)

# One-hot membership
membership_levels = np.unique(membership)
membership_enc = np.zeros((len(membership), len(membership_levels)))
for i, m in enumerate(membership_levels):
    membership_enc[:, i] = (membership == m).astype(float)

# Standardize numeric features
numeric = np.column_stack((age, total_spend, items, rating, days))
numeric = (numeric - numeric.mean(axis=0)) / numeric.std(axis=0)

X = np.column_stack([
    gender_enc,
    city_enc,
    membership_enc,
    discount_enc,
    numeric
])

def train_test_split(X, y, test_size=0.25, seed=6769):
    np.random.seed(seed)
    idx = np.random.permutation(len(X))
    test_count = int(len(X) * test_size)
    test_idx = idx[:test_count]
    train_idx = idx[test_count:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

X_train, X_test, y_train, y_test = train_test_split(X, y)

clf = LogisticRegression()
clf.fit(X_train, y_train)

preds = clf.predict(X_test)
acc = np.mean(preds == y_test)
print("Accuracy:", acc)
print("First 10 preds:", preds[:10])
print("First 10 true:", y_test[:10])

def plot_feature_importance(clf, feature_names, top_n=None):
    weights = clf.weights

    # Sort by magnitude
    sorted_idx = np.argsort(np.abs(weights))[::-1]

    if top_n:
        sorted_idx = sorted_idx[:top_n]

    plt.figure(figsize=(10, 6))
    plt.barh(
        [feature_names[i] for i in sorted_idx],
        weights[sorted_idx],
        color="royalblue"
    )
    plt.xlabel("Weight")
    plt.title("Logistic Regression Feature Importance")
    plt.gca().invert_yaxis()
    plt.show()

def plot_probability_curve(clf, X, y, feature_idx, feature_name):
    x_feat = X[:, feature_idx]

    # Grid for smooth line
    xs = np.linspace(x_feat.min(), x_feat.max(), 300)

    # Use the mean of all other features
    X_mean = X.mean(axis=0)
    X_grid = np.tile(X_mean, (300, 1))
    X_grid[:, feature_idx] = xs

    # Compute probabilities
    z = X_grid @ clf.weights + clf.bias
    probs = 1 / (1 + np.exp(-z))

    plt.figure(figsize=(8, 5))

    # True labels (jitter so you can see them)
    jitter = np.random.normal(scale=0.02, size=len(x_feat))
    plt.scatter(x_feat, y + jitter, alpha=0.3, label="Actual Labels")

    # Probability curve
    plt.plot(xs, probs, color="red", linewidth=2, label="Predicted Probability")

    plt.xlabel(feature_name)
    plt.ylabel("Probability of Satisfaction")
    plt.title(f"Model Behavior Along Feature: {feature_name}")
    plt.legend()
    plt.show()

feature_names = ["Male", "Female"]
feature_names += [f"City: {c}" for c in cities]
feature_names += [f"Membership: {m}" for m in membership_levels]
feature_names += ["DiscountApplied"]
feature_names += ["Age", "TotalSpend", "ItemsPurchased", "Rating", "DaysSinceLastPurchase"]

def plot_pred_vs_actual(clf, X_test, y_test):
    linear = X_test @ clf.weights + clf.bias
    probs = 1 / (1 + np.exp(-linear))

    plt.figure(figsize=(7, 5))
    
    plt.scatter(range(len(probs)), probs, 
                label="Predicted Probabilities", alpha=0.5)
    
    plt.scatter(range(len(y_test)), y_test, 
                label="Actual Labels", alpha=0.5)
    
    plt.axhline(0.5, color='red', linestyle='--', linewidth=2, 
                label="Decision Threshold (0.5)")

    plt.legend()
    plt.title("Predicted Satisfaction vs. Actual Labels")
    plt.xlabel("Test Sample Index")
    plt.ylabel("Probability / Label")
    plt.show()


plot_pred_vs_actual(clf, X_test, y_test)
plot_feature_importance(clf, feature_names)
plot_probability_curve(clf, X, y, feature_idx= feature_names.index("TotalSpend"), feature_name="TotalSpend")