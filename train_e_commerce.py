import numpy as np
from LogisticRegression import LogisticRegression
from visualization import plot_pred_vs_actual, plot_feature_importance, plot_probability_curve

data = np.genfromtxt("customers.csv", delimiter=",", dtype=str, skip_header=1)

# Labels
y_raw = data[:, -1]

# Satisfied = 1, else = 0
y = (y_raw == "Satisfied").astype(float)

# Features =====================================================================================
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

gender_enc = np.where(gender == "Male", 1, 0).reshape(-1, 1)
discount_enc = np.where(discount == "TRUE", 1, 0).reshape(-1, 1)

cities = np.unique(city)
city_enc = np.zeros((len(city), len(cities)))
for i, c in enumerate(cities):
    city_enc[:, i] = (city == c).astype(float)

membership_levels = np.unique(membership)
membership_enc = np.zeros((len(membership), len(membership_levels)))
for i, m in enumerate(membership_levels):
    membership_enc[:, i] = (membership == m).astype(float)

numeric = np.column_stack((age, total_spend, items, rating, days))
numeric = (numeric - numeric.mean(axis=0)) / numeric.std(axis=0)

X = np.column_stack([
    gender_enc,
    city_enc,
    membership_enc,
    discount_enc,
    numeric
])
# ===============================================================================================

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


feature_names = ["Male", "Female"]
feature_names += [f"City: {c}" for c in cities]
feature_names += [f"Membership: {m}" for m in membership_levels]
feature_names += ["DiscountApplied"]
feature_names += ["Age", "TotalSpend", "ItemsPurchased", "Rating", "DaysSinceLastPurchase"]

# Visualizations
plot_pred_vs_actual(clf, X_test, y_test)
plot_feature_importance(clf, feature_names)
plot_probability_curve(clf, X, y, feature_idx= feature_names.index("DiscountApplied"), feature_name="DiscountApplied")

X_train = clf.recursive_feature_elimination(X_train, y_train, feature_names, top_n=5)
print("Top 5 features:", X_train)
clf.fit(X_train, y_train)

preds = clf.predict(X_test)
acc = np.mean(preds == y_test)
print("Accuracy:", acc)