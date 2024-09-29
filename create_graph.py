import numpy as np

e1 = [1] * 5 + [0] * 4
e2 = [0] * 5 + [1] * 4
e1 = np.array(e1)
e2 = np.array(e2)

n = e1.shape[0]

A = e1[:, None] @ e1[:, None].T + e2[:, None] @ e2[:, None].T
A[np.eye(n, dtype=bool)] = 0
print(A)

Ri = []
for i in range(2):
    Ux = np.random.randn(n, 2)
    s = np.array([2, 1])
    R = Ux @ np.diag(s) @ Ux.T
    R += 0.1 * np.random.randn(n, n)
    Ri.append(R)
    # print(R)


test