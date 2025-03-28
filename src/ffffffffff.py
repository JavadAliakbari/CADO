# from copy import deepcopy

# import numpy as np
# from sklearn.metrics.cluster import adjusted_rand_score
# from tqdm import tqdm

# from utils.SBM import SBM
# from utils.utils import *

# if __name__ == "__main__":
#     samples_per_cluster = config.samples_per_cluster
#     d = config.num_features
#     na = config.num_signal_features
#     n = sum(samples_per_cluster)
#     c = len(samples_per_cluster)
#     n_clusters = c

#     probs = np.full((c, c), config.q)
#     np.fill_diagonal(probs, config.p)
#     config.tau = n
#     graph = SBM(
#         samples_per_cluster,
#         probs,
#         d,
#         na,
#         # train_ratio=1,
#         train_ratio=config.train_ratio,
#     )
#     plot_adj(graph.A, name="A")

#     Rv_matrices = deepcopy(graph.Rv_matrices)
#     Ri_opt = graph.Ri_matrices
#     E = np.zeros((n, na))

#     Z = np.zeros((n, d, d))
#     lam = n * np.random.rand((na))
#     Ri = np.zeros((na, d, d))

#     rho = 1.0
#     iter_ALM = 2

#     for iter in range(config.cogent_iters):

#         Rv_estim = np.einsum("ij,jkn->ikn", ((E**2) @ np.diag(lam)), Ri)
#         Rv_error = Rv_matrices - Rv_estim

#         # update Z
#         Z += rho * Rv_error

#         # update Rv
#         Rv_inv = np.array([invert_matrix(matrix) for matrix in Rv_matrices])
#         grad_Rv = np.eye(d) - np.einsum("ijk,ikm,imn->ijn", Rv_inv, graph.cov, Rv_inv)
#         Rv_matrices = Rv_estim + (grad_Rv - Z) / rho

#         M = Rv_error + Z / rho
#         for i in range(na):
#             e_temp = E[:, i]
#             lam_temp = lam[i]
#             M += lam_temp * np.einsum("i,jk->ijk", e_temp**2, Ri[i])
#             for t in range(iter_ALM):

#                 # update Ri
#                 numerator = np.einsum("v,vij->ij", e_temp**2, Rv_matrices)
#                 denominator = lam_temp 
#                 Ri[i] = numerator / denominator
#                 Ri[i] = project_to_psd_cone(Ri[i], 0, config.rho_max - config.rho_min)

#                 # update lam_i
#                 numerator = np.sum(epsilon_squared * np.einsum('vij,ij->v', R, X)) - (mu / 2)
#                 denominator = np.linalg.norm(X, 'fro')**2 * np.sum(epsilon_fourth)
#                 lam[i] = numerator / denominator

#                 # update e_i
#                 # E[:, i]

#             M -= lam_temp * np.einsum("i,jk->ijk", e_temp**2, Ri[i])

#         print(f"Iteration {iter}")
#         print(f"Norm of Rv_error: {np.linalg.norm(Rv_error)}")



import pandas as pd

# Creating a 4x4 numerical dataframe
data = {
    "A": [1, 2, 3, 4],
    "B": [5, 6, 7, 8],
    "C": [9, 10, 11, 12],
    "D": [13, 14, 15, 16]
}

df = pd.DataFrame(data)

# Save the dataframe as a CSV file
file_path = "/mnt/data/numerical_data.csv"
df.to_csv(file_path, index=False)

# Provide the file for download
file_path
