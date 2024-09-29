import matplotlib.pyplot as plt
import numpy as np


def plot_decision_boundary(model, X, y, ax, plot_title="Decision Boundary"):
    """
    Plots the decision boundary of a fitted sklearn model for 2D data on the given axes.

    Parameters:
    - model: a fitted scikit-learn model
    - X: 2D array-like, shape (n_samples, 2), the input data with two features
    - y: array-like, shape (n_samples,), the class labels
    - ax: matplotlib.axes.Axes, the axes on which to plot
    - plot_title: string, optional (default='Decision Boundary'), title for the plot
    """
    # Ensure that the input data has two features
    assert X.shape[1] == 2, "X must have exactly 2 features to plot decision boundary."

    # Create a mesh grid that covers the entire feature space
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

    # Predict class labels for each point in the grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the decision boundary and data points on the given axes
    ax.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)

    # Plot the original data points
    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k", cmap=plt.cm.coolwarm)

    # Labels and titles
    ax.set_title(plot_title)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
