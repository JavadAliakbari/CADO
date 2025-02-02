from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm

from utils.utils import *


def EM_step(R, Ri):
    R_inv = np.linalg.inv(R)
    zz = R_inv @ Ri @ R_inv
    # P = np.eye(R.shape[0]) - zz
    P = R_inv - zz
    S, U = np.linalg.eigh(P)

    mask1 = S < 0

    S2 = deepcopy(S)
    S2[mask1] = config.rho_max
    S2[~mask1] = config.rho_min
    RR = U @ np.diag(S2) @ U.T

    l = matrix_dot_product(P, RR)
    # l = RR[0, 0]

    return RR, l


n = 100000
num_features = 6
num_signal_features = 2


Ri = create_Ri_symmetric_matrix_with_constrained_eigenvalues2(
    num_features,
    config.sig_rho_min,
    config.sig_rho_max,
    num_signal_features,
)
Ss, Us = np.linalg.eigh(Ri)
print(np.round(Us, 2))

X = np.random.multivariate_normal(mean=np.zeros(num_features), cov=Ri, size=n)

R0 = create_Ri_symmetric_matrix_with_constrained_eigenvalues(
    num_features,
    config.rho_min,
    config.rho_max,
)

Rih = np.einsum("ij,ik->jk", X, X) / n
# R0 = project_to_set_S(R0, config.rho_min, config.rho_max)
# print(random.seed)

total = 2500
bar = tqdm(total=total)
l_list = []
for i in range(total):
    II = np.eye(R0.shape[0])
    R0, l = EM_step(II, Rih)

    bar.set_postfix_str(l)

    bar.update()
    l_list.append(l)

print(Ri)

print(np.round(R0, 2))

_, ll = EM_step(Ri, Rih)
# ll = calc_ML(X, Ri)
print(ll)
plt.plot(l_list - l_list[-1])
plt.yscale("log")
plt.show()
