from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm

from utils.utils import *


def create_P(R, RR):
    R_inv = np.linalg.inv(R)
    zz = R_inv @ RR @ R_inv
    P = np.eye(R.shape[0]) - zz
    # P = R_inv - zz
    return P


def EM_step(R, RR):
    P = create_P(R, RR)

    Rp = R - 0.01 * P
    Rp = project_to_psd_cone(Rp, config.rho_min, config.rho_max)

    return Rp


n = 100000
num_features = 6
num_signal_features = 2


Ri = create_Ri_symmetric_matrix_with_constrained_eigenvalues2(
    num_features,
    config.sig_rho_min,
    config.sig_rho_max,
    num_signal_features,
)

X = np.random.multivariate_normal(mean=np.zeros(num_features), cov=Ri, size=n)

R0 = create_Ri_symmetric_matrix_with_constrained_eigenvalues(
    num_features,
    config.rho_min,
    config.rho_max,
)

RR = np.einsum("ij,ik->jk", X, X) / n
print(np.round(RR, 2))

total = 1000
bar = tqdm(total=total)

l_list = []
for i in range(total):
    R0 = EM_step(R0, RR)

    R_inv = np.linalg.inv(R0)
    # l = np.trace(RR @ R_inv) + np.log(np.linalg.det(R0))
    l = np.trace(RR @ R_inv) + np.trace(R0)

    bar.set_postfix_str(l)

    bar.update()
    l_list.append(l)

print(Ri)

print(np.round(R0, 2))

R_inv = np.linalg.inv(Ri)
# ll = np.trace(RR @ R_inv) + np.log(np.linalg.det(Ri))
ll = np.trace(RR @ R_inv) + np.trace(Ri)
print(ll)
# plt.plot(l_list)
plt.plot(l_list - l_list[-1])
plt.yscale("log")
plt.show()
