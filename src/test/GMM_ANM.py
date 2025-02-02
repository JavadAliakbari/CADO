import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from numpy.linalg import norm, det, inv


# Helper function: Multivariate Gaussian PDF
def multivariate_gaussian(x, mean, cov):
    n = x.shape[1]  # Dimensionality
    diff = x - mean
    exp_part = np.exp(-0.5 * np.sum(diff @ inv(cov) * diff, axis=1))
    coeff = 1.0 / np.sqrt((2 * np.pi) ** n * det(cov))
    return coeff * exp_part


# Negative Log-Likelihood (NLL) with regularization
def nll_atomic_norm(X, weights, means, covariances, lambd):
    n_samples, n_features = X.shape
    n_components = len(weights)

    # Compute NLL
    nll = 0
    for i in range(n_samples):
        likelihood = 0
        for k in range(n_components):
            likelihood += weights[k] * multivariate_gaussian(
                X[i : i + 1], means[k], covariances[k]
            )
        nll -= np.log(likelihood)

    # Atomic norm minimization (l1-norm of the means for sparsity)
    atomic_norm = lambd * np.sum([norm(mu, ord=1) for mu in means])

    return nll + atomic_norm


# Fit GMM with Atomic Norm Regularization using scipy.optimize
def fit_gmm_with_anm(X, n_components, lambd, n_iters=100):
    n_samples, n_features = X.shape

    # Initialize parameters (weights, means, covariances)
    weights = np.ones(n_components) / n_components
    means = X[np.random.choice(n_samples, n_components, replace=False)]
    covariances = np.array([np.eye(n_features)] * n_components)

    def objective(params):
        weights = params[:n_components]
        means = params[n_components : n_components * (1 + n_features)].reshape(
            n_components, n_features
        )
        covariances = np.array(
            [np.eye(n_features)] * n_components
        )  # Fixed covariances for simplicity
        return nll_atomic_norm(X, weights, means, covariances, lambd)

    # Pack initial parameters
    initial_params = np.hstack([weights, means.flatten()])

    # Use scipy's minimize function to optimize the objective
    result = minimize(
        objective, initial_params, method="L-BFGS-B", options={"maxiter": n_iters}
    )

    # Unpack the optimized parameters
    optimized_params = result.x
    weights_opt = optimized_params[:n_components]
    means_opt = optimized_params[
        n_components : n_components * (1 + n_features)
    ].reshape(n_components, n_features)

    return weights_opt, means_opt, covariances


# Generate synthetic data
np.random.seed(42)
X1 = np.random.normal(loc=0.0, scale=1.0, size=(200, 2))  # Cluster 1
X2 = np.random.normal(loc=5.0, scale=1.0, size=(200, 2))  # Cluster 2
X = np.vstack((X1, X2))

# Fit GMM with Atomic Norm Minimization
n_components = 2
lambd = 0.1  # Regularization parameter for atomic norm
weights, means, covariances = fit_gmm_with_anm(X, n_components, lambd)

# Predict cluster labels
labels = np.argmax(
    [multivariate_gaussian(X, means[k], covariances[k]) for k in range(n_components)],
    axis=0,
)

# Visualize the clustering result
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis")
plt.title("Clusters Identified with GMM + Atomic Norm")
plt.show()

# Output the optimized means and weights
print("Optimized Weights:", weights)
print("Optimized Means:\n", means)
