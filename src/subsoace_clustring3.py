from matplotlib import pyplot as plt
from sklearn.metrics import adjusted_rand_score
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from subspace_clustring import visualize_high_dim_data

# --- Settings and Data Generation ---

# Device setting: use GPU if available.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data parameters
n_samples = 3000  # total number of data points
n_features = 10  # ambient dimension
k = 3  # number of clusters
d = 2  # subspace dimension (d > 1)

# Known noise and signal scales:
sigma_noise = 0.2
sigma_signal = 5.0

# Generate synthetic data.
# All clusters share the same mean (zero) and same noise and signal power.
# They differ only in the subspace (signal part).
X_list = []
true_labels = []
samples_per_cluster = n_samples // k
for i in range(k):
    # Create a random subspace basis (n_features x d) and orthonormalize it.
    A = np.random.randn(n_features, d)
    Q, _ = np.linalg.qr(A)
    U_true = Q[:, :d]
    # Generate low-dimensional coefficients (the signal)
    s = sigma_signal * np.random.randn(samples_per_cluster, d)
    signal = s @ U_true.T
    # Generate additive noise
    noise = sigma_noise * np.random.randn(samples_per_cluster, n_features)
    X_cluster = signal + noise
    X_list.append(X_cluster)
    true_labels.extend([i] * samples_per_cluster)

X_np = np.vstack(X_list)
true_labels = np.array(true_labels)

# Convert data to a torch tensor.
X = torch.tensor(
    X_np, dtype=torch.float32, device=device
)  # shape: (n_samples, n_features)

# --- Define Parameters for Gradient Descent ---

# We parameterize the soft assignments with free parameters alpha (n_samples x k).
alpha_param = nn.Parameter(torch.randn(n_samples, k, device=device))

# We parameterize each cluster's subspace U (k x n_features x d).
# Each U[i] is the basis for cluster i.
U_param = nn.Parameter(torch.randn(k, n_features, d, device=device))

# Define an optimizer for both sets of parameters.
optimizer = optim.Adam([alpha_param, U_param], lr=1e-2)

# --- Helper Function: Reproject U onto the Stiefel Manifold ---


def project_onto_stiefel(U):
    """
    Given a matrix U (n_features x d), perform a QR decomposition
    and return the Q factor (which has orthonormal columns).
    """
    Q, _ = torch.qr(U)
    return Q


# --- Gradient Descent Loop ---

num_epochs = 5000
I = torch.eye(n_features, device=device)  # Identity matrix for later use

for epoch in range(num_epochs):
    optimizer.zero_grad()

    # Compute soft assignments p from the free parameters alpha:
    # p has shape (n_samples, k) and each row sums to 1.
    p = torch.softmax(alpha_param, dim=1)  # p[v, i] = p_{iv}

    # For each cluster i, form R_i = sigma_noise^2 I + sigma_signal^2 U_i U_i^T.
    # U_param has shape (k, n_features, d)
    # Compute U U^T for each cluster:
    R_list = sigma_noise**2 * I.unsqueeze(0).expand(
        k, n_features, n_features
    ) + sigma_signal**2 * torch.bmm(
        U_param, U_param.transpose(1, 2)
    )  # shape: (k, n_features, n_features)

    # For each data point v, compute R_v = sum_i p_{iv} R_i.
    # p has shape (n_samples, k); we want to compute a weighted sum for each v.
    p_expanded = p.view(n_samples, k, 1, 1)  # shape: (n_samples, k, 1, 1)
    R_list_expanded = R_list.unsqueeze(0)  # shape: (1, k, n_features, n_features)
    R_v = torch.sum(
        p_expanded * R_list_expanded, dim=1
    )  # shape: (n_samples, n_features, n_features)

    # Now, compute the cost for all data points.
    # For each v, compute: log(det(R_v)) + x_v^T R_v^{-1} x_v.
    cost = 0
    for v in range(n_samples):
        x_v = X[v].view(n_features, 1)  # (n_features, 1)
        R_v_mat = R_v[v]
        # For numerical stability, if needed add a small jitter.
        sign, logdet = torch.slogdet(R_v_mat)
        if sign.item() <= 0:
            R_v_mat = R_v_mat + 1e-6 * I
            sign, logdet = torch.slogdet(R_v_mat)
        R_v_inv = torch.inverse(R_v_mat)
        quad = torch.matmul(x_v.t(), torch.matmul(R_v_inv, x_v))
        cost = cost + logdet + quad.squeeze()

    cost = cost / n_samples  # average cost per data point

    # Backpropagate.
    cost.backward()
    optimizer.step()

    # Project each U_i onto the Stiefel manifold (reorthonormalize its columns)
    with torch.no_grad():
        for i in range(k):
            U_param[i].copy_(project_onto_stiefel(U_param[i]))

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, cost: {cost.item():.5f}")

print("Optimization finished.")

clusters = np.argmax(p, axis=1)

# Measure ARI performance.
ari = adjusted_rand_score(true_labels, clusters)
print("Adjusted Rand Index (ARI):", ari)

# Visualize using PCA to project to 3 dimensions.
visualize_high_dim_data(X, true_labels, U_true, np.zeros((k, n_features)), n_features)
visualize_high_dim_data(X, clusters, U_param, np.zeros((k, n_features)), n_features)

plt.show()
