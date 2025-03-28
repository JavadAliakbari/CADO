import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import qr

class_num = 3
samp_per_class = 100
samp_total = class_num * samp_per_class
num_atom = 3

p_flip = 0.1


rho = 0.1
step = 0.1
reg = 0.01 * samp_total
iter_num = 400

Graph_ideal = -np.ones((samp_total, samp_total))

for count in range(class_num):
    Graph_ideal[
        count * samp_per_class : (count + 1) * samp_per_class,
        count * samp_per_class : (count + 1) * samp_per_class,
    ] = 1

flip_pattern = 1 - 2 * np.random.binomial(1, p_flip, (samp_total, samp_total))

Graph = Graph_ideal * flip_pattern

L = Graph
Z = np.zeros((samp_total, samp_total))

E = np.zeros((samp_total, num_atom))
lam = np.zeros((num_atom))

e_temp = np.zeros((samp_total))

for count in range(iter_num):
    Estim = E @ np.diag(lam) @ E.T
    Error = L - Estim

    Z += rho * Error

    val1 = np.linalg.norm(Error)
    val2 = reg * np.sum(lam) - np.sum(Graph * L)

    print(val1, val2)

    # L+=step*(Graph-Z)-Error
    L = (rho * Estim - Z + Graph) / rho
    L[L > 1] = 1
    L[L < 0] = 0

    M = Error + Z / rho
    for count2 in range(num_atom):
        e_temp = E[:, count2]
        M += lam[count2] * np.outer(e_temp, e_temp)
        lam_comp, e_comp = np.linalg.eigh(M)
        lam_temp = lam_comp[-1] - reg / rho
        if lam_temp < 0:
            lam_temp = 0
        e_temp = e_comp[:, -1]
        lam[count2] = lam_temp
        E[:, count2] = e_temp
        M -= lam_temp * np.outer(e_temp, e_temp)

print(lam)

plt.figure()
for i in range(num_atom):
    plt.plot(E[:, i])
plt.show()

plt.figure()
plt.imshow(Estim)
plt.colorbar()
plt.show()

