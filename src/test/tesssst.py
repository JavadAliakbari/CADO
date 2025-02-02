import numpy as np
from tqdm import tqdm

from utils.utils import *

# E = np.array(
#     [
#         [1 / 4, 0, 0],
#         [1 / 4, 0, 0],
#         [1 / 4, 0, 0],
#         [1 / 4, 0, 0],
#         [0, 1 / 3, 0],
#         [0, 1 / 3, 0],
#         [0, 1 / 3, 0],
#         [0, 0, 1 / 2],
#         [0, 0, 1 / 2],
#     ]
# )
# H = np.array(
#     [
#         [1 / 3, 1 / 2, 1 / 6],
#         [1 / 4, 1 / 4, 1 / 2],
#         [1 / 5, 1 / 6, 19 / 30],
#         [1 / 3, 1 / 3, 1 / 3],
#         [1 / 2, 1 / 2, 0],
#     ]
# )
H = np.random.randn(9, 3)
H = project_to_simplex2(H, 1)
true_l = np.sum(H, axis=0)
E = H @ np.diag(1 / true_l)
n, c = E.shape


# l = n * np.random.rand(c)
l = n * np.ones(c) / c
l = project_to_simplex2(l, n)

total = 1000
bar = tqdm(total=total)
for i in range(total):
    H = E @ np.diag(l)
    A = project_to_simplex2(H, 1)
    new_l = np.sum(A, axis=0)

    f1 = sum(sum(np.abs(H - A)))
    f2 = sum(np.abs(new_l - l))
    l = new_l

    bar.set_postfix(f1=f1, f2=f2)
    bar.update()

print(l)
print(true_l)
print(np.round(A, 2))
