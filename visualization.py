from matplotlib import pyplot as plt
import numpy as np

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