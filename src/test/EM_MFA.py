import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import multivariate_normal as mvn
import os
from os.path import join as pjoin

np.random.seed(13)

K = 3
n = 100
D = 2
d = 1
X = np.zeros((n * K, D))

# Loadings
Lambda = np.random.normal(scale=5, size=(K, D, d + 1))

# Noise covariance
Psi = np.diag(np.ones(D))

# Latent variables
z = np.random.normal(size=(d, n * K))
z = np.concatenate([z, np.ones((1, n * K))])

mvn_gen = mvn(mean=np.zeros(D), cov=Psi)

for k in range(K):

    # Generate data
    k_idx = np.arange(k * n, (k + 1) * n)
    curr_data = (Lambda[k] @ z[:, k_idx]).T + mvn_gen.rvs(n)
    X[k_idx] = curr_data


X = X.T

## Fit EM

Lambda_est = np.random.normal(scale=10, size=(K, D, d + 1))
Psi_est = np.diag(np.ones(D))
pi = np.ones(K) / K
h = np.ones((K, n * K))
n_iters = 40

for ii in range(n_iters):

    # Compute h
    for k in range(K):

        L = Lambda_est[k][:, :d]
        mu = Lambda_est[k][:, d:]

        h[k, :] = pi[k] * mvn(mean=mu.squeeze(), cov=L @ L.T + Psi_est).pdf(X.T)

    h = h / np.sum(h, axis=0)

    Psi_new = np.zeros((D, D))
    for k in range(K):

        # Compute beta
        L = Lambda_est[k][:, :d]
        mu = Lambda_est[k][:, d:]
        X_centered = X - mu
        beta = L.T @ np.linalg.inv(Psi_est + L @ L.T)

        # Compute expectations
        Ez = beta @ X_centered
        Ez = np.concatenate([Ez, np.ones((1, n * K))])

        EzzT = np.zeros((d + 1, d + 1))
        EzzT[:d, :d] = (
            np.sum(h[k]) * (np.eye(d) - beta @ L)
            + beta @ (h[k] * X_centered) @ X_centered.T @ beta.T
        )

        EzzT_offdiag_term = np.sum(h[k] * Ez[:d, :], axis=1)
        EzzT[:d, d:] = EzzT_offdiag_term
        EzzT[d:, :d] = EzzT_offdiag_term.T
        EzzT[d:, d:] = np.sum(h[k])

        # Update parameters
        Lambda_est[k] = (h[k] * X) @ Ez.T @ np.linalg.inv(EzzT)
        Psi_new += np.diag(np.diag((X - Lambda_est[k] @ Ez) @ ((h[k] * X).T)))

    Psi_est = Psi_new / (n * K)

    # Update pi
    pi = np.mean(h, axis=1)
    print(pi)
