import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture


# Helper function to calculate multivariate Gaussian PDF
def multivariate_gaussian(x, mean, cov):
    n = x.shape[1]  # Dimensionality
    diff = x - mean
    exp_part = np.exp(-0.5 * np.sum(diff @ np.linalg.inv(cov) * diff, axis=1))
    coeff = 1.0 / np.sqrt((2 * np.pi) ** n * np.linalg.det(cov))
    return coeff * exp_part


# Gaussian Mixture Model (GMM) class implementation
class GMM:
    def __init__(self, n_components, n_iters=100, tol=1e-6):
        self.n_components = n_components  # Number of Gaussian components (clusters)
        self.n_iters = n_iters  # Number of iterations for EM
        self.tol = tol  # Convergence tolerance
        self.weights = None  # Mixing coefficients
        self.means = None  # Mean of each Gaussian component
        self.covariances = None  # Covariance of each Gaussian component

    def fit(self, X):
        # Initialize parameters
        n_samples, n_features = X.shape
        self.weights = np.ones(self.n_components) / self.n_components
        self.means = X[np.random.choice(n_samples, self.n_components, replace=False)]
        self.covariances = np.array([np.eye(n_features)] * self.n_components)
        log_likelihood = 0

        # EM algorithm
        for i in range(self.n_iters):
            # E-step: Calculate responsibilities (gamma) for each data point
            responsibilities = np.zeros((n_samples, self.n_components))
            for k in range(self.n_components):
                responsibilities[:, k] = self.weights[k] * multivariate_gaussian(
                    X, self.means[k], self.covariances[k]
                )
            responsibilities /= responsibilities.sum(axis=1, keepdims=True)

            # M-step: Update weights, means, and covariances
            Nk = responsibilities.sum(
                axis=0
            )  # Effective number of points assigned to each Gaussian
            self.weights = Nk / n_samples
            self.means = np.dot(responsibilities.T, X) / Nk[:, np.newaxis]
            for k in range(self.n_components):
                diff = X - self.means[k]
                self.covariances[k] = (
                    np.dot(responsibilities[:, k] * diff.T, diff) / Nk[k]
                )

            # Check for convergence (log-likelihood)
            new_log_likelihood = np.sum(np.log(np.sum(responsibilities, axis=1)))
            if np.abs(new_log_likelihood - log_likelihood) < self.tol:
                break
            log_likelihood = new_log_likelihood

    def predict(self, X):
        responsibilities = np.zeros((X.shape[0], self.n_components))
        for k in range(self.n_components):
            responsibilities[:, k] = self.weights[k] * multivariate_gaussian(
                X, self.means[k], self.covariances[k]
            )
        return responsibilities.argmax(axis=1)  # Cluster labels


# Generate synthetic data
# np.random.seed(42)
X1 = np.random.multivariate_normal(
    mean=[0.0, 0.0], cov=[[1.0, 0], [0, 0.05]], size=(20000)
)  # Cluster 1
X2 = np.random.multivariate_normal(
    mean=[0.0, 0.0], cov=[[0.05, 0.0], [0.0, 1.0]], size=(20000)
)  # Cluster 2
# X1 = np.random.normal(loc=0.0, scale=1.0, size=(200, 2))  # Cluster 2
# X2 = np.random.normal(loc=5.0, scale=1.0, size=(200, 2))  # Cluster 2
X = np.vstack((X1, X2))

# Visualize the synthetic data
plt.scatter(X1[:, 0], X1[:, 1])
plt.scatter(X2[:, 0], X2[:, 1])
# plt.scatter(X[:, 0], X[:, 1])
plt.title("Generated Data")
# plt.show()

# # Implementing GMM
# gmm = GaussianMixture(n_components=2, covariance_type='full', init_params="random")

# # Fitting the model
# gmm.fit(X)

# # Predicting the cluster labels
# labels = gmm.predict(X)

# Apply GMM
gmm = GMM(n_components=2, n_iters=100)
gmm.fit(X)

# Predict the clusters
labels = gmm.predict(X)

# Visualize the clusters
plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis")
plt.title("Clusters Identified by GMM")
plt.show()

# Output the means and covariances of the Gaussian components
print("Means of each component:\n", gmm.means)
print("\nCovariances of each component:\n", gmm.covariances)
