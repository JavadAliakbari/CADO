import numpy as np

e1 = [1] * 5 + [0] * 4
e2 = [0] * 5 + [1] * 4
e1 = np.array(e1)
e2 = np.array(e2)

A = e1[:, None] @ e1[:, None].T + e2[:, None] @ e2[:, None].T
A[np.eye(A.shape[0], dtype=bool)] = 0
print(A)
