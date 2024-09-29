import numpy as np


p1 = [0, 1]
p2 = [1, 0]
ni = [5, 4]
C = len(ni)
e1 = [1] * ni[0] + [0] * ni[1]
e2 = [0] * ni[0] + [1] * ni[1]
e1 = np.array(e1)
e2 = np.array(e2)

n = e1.shape[0]

A = e1[:, None] @ e1[:, None].T + e2[:, None] @ e2[:, None].T
# A[np.eye(n, dtype=bool)] = 0
S, E = np.linalg.eigh(A)
S = S[-C:]
E = E[:, -C:]
print(E)

Ri = []
d = 5
for i in range(C):
    Ux = np.random.randn(d, 2)
    s = np.array([2, 1])
    R = Ux @ np.diag(s) @ Ux.T
    R += 0.1 * np.random.randn(d, d)
    Ri.append(R)
    # print(R)
Ri = np.stack(Ri, axis=0)

Rvs = np.einsum("ij,jkn->ikn", ((E**2) @ np.diag(S)), Ri)

X = []
for Rv in Rvs:
    Xv = np.random.multivariate_normal(np.zeros(d), Rv, 1)[0]
    X.append(Xv)

X = np.array(X)

# p1 = [0.3, 0.7]
# p2 = [0.8, 0.2]

b = np.stack([p1, p2], axis=0)
# b = np.stack([e2, e1], axis=1)
pi = ((E**2) @ np.diag(S)) @ b

y = []
for pi_v in pi:
    yi = np.random.choice(C, 1, p=pi_v)[0]
    y.append(yi)

y = np.array(y)
print(y)
