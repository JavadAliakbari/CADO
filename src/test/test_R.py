import numpy as np
from utils.utils import *

n = 6
d = 4
rho_min = 1
rho_max = 2

eps = np.random.randn(n)
z = np.random.randn(n, d)

Pv = np.zeros((d, d))
# Pv2 = np.zeros(R.shape)
for j in range(n):
    Pv += eps[j] ** 2 * (-np.outer(z[j], z[j]) + np.eye(d))
    # Pv += Pv2 + eps[j]**2 * np.eye(d)

S, U = np.linalg.eigh(Pv)
# print(S)
# S[S<rho_min] = rho_min
# S[S>rho_max] = rho_max
S[S > 0] = rho_max
S[S < 0] = rho_min
Rhat = U @ np.diag(S) @ U.T

step = 0.01
R = np.random.randn(d, d)
for i in range(30000):
    tmp = R + step * Pv
    R_new = project_to_psd_cone(tmp, rho_min, rho_max)
    th = np.linalg.norm(R_new - R)
    R = R_new
    if th < 0.0000001:
        break

S2, Q = np.linalg.eigh(R)

print(np.sum(Pv * R))
print(np.sum(Pv * Rhat))

print(S)
print(S2)
print(Q)
print(U)
# print(th)
# print(i)
