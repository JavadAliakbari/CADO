from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm

from utils.utils import *


def create_P(R, RR):
    Rv = R + config.rho_min * np.eye(R.shape[0])
    R_inv = np.linalg.inv(Rv)
    zz = R_inv @ RR @ R_inv
    P = np.eye(R.shape[0]) - zz
    # P = R_inv - zz
    return P


def EM_step(R, RR):
    P = create_P(R, RR)
    # P = R_inv - np.mean(Pv, axis=0)
    S, U = np.linalg.eigh(P)

    mask1 = S < 0

    S2 = deepcopy(S)
    S2[mask1] = config.rho_max - config.rho_min
    S2[~mask1] = 0
    Rp = U @ np.diag(S2) @ U.T + config.rho_min * np.eye(R.shape[0])

    l = matrix_dot_product(P, Rp)
    # l = RR[0, 0]

    # Rp -= config.rho_min * np.eye(Rp.shape[0])

    return Rp, l


n = 10000
num_features = 6
num_signal_features = 2


Ri = create_Ri_symmetric_matrix_with_constrained_eigenvalues2(
    num_features,
    config.sig_rho_min,
    config.sig_rho_max,
    num_signal_features,
)
# Ss, Us = np.linalg.eigh(Ri)
# print(np.round(Us, 2))
print(np.round(Ri, 2))

X = np.random.multivariate_normal(mean=np.zeros(num_features), cov=Ri, size=n)

R0 = create_Ri_symmetric_matrix_with_constrained_eigenvalues(
    num_features,
    config.rho_min,
    config.rho_max,
)

RR = np.einsum("ij,ik->jk", X, X) / n - config.rho_min * np.eye(num_features)
# R0 = project_to_set_S(R0, config.rho_min, config.rho_max)
# print(random.seed)
# print(np.round(Ri, 2))

total = 1
bar = tqdm(total=total)
l_list = []
R0 = np.zeros(R0.shape[0])
for i in range(total):
    R0, l = EM_step(R0, RR)

    R_inv = np.linalg.inv(R0)
    # l = np.trace(RR @ R_inv) + np.log(np.linalg.det(R0))
    l = np.trace(RR @ R_inv) + np.trace(R0)

    bar.set_postfix_str(l)

    bar.update()
    l_list.append(l)


print(np.round(R0, 2))

_, ll = EM_step(Ri, RR)
# ll = calc_ML(X, Ri)
print(ll)
# plt.plot(l_list - l_list[-1])
# plt.yscale("log")
# plt.show()
