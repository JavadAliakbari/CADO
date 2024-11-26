import pickle as pkl

import numpy as np

from Atom.atoms import Atoms
from Atom.coefs import Coefs
from Graph_Atom.graph_atom import GraphAtom
from Graph_Atom.graph_func import GraphFunc
from utils.utils import *

path = "results/20241107_122558/"

atoms_path = f"{path}atoms.pkl"
coefs_path = f"{path}coefs.pkl"
graph_path = f"{path}graph.pkl"

with open(atoms_path, "rb") as fid:
    atoms = pkl.load(fid)
with open(coefs_path, "rb") as fid:
    coefs = pkl.load(fid)
with open(graph_path, "rb") as fid:
    graph = pkl.load(fid)


f = GraphFunc(graph=graph)

sig = atoms.calc_sig(coefs)
f_sig = f(sig)
print(f_sig)


## Initialization
atoms = Atoms()
coefs = Coefs()

E = graph.E
# E[np.abs(E) < 0.00001] = 0.00001
R = graph.Ri_matrices
b = graph.b
c = graph.num_clusters

for i in range(c):
    atom0 = GraphAtom()
    S, U = np.linalg.eigh(R[i])
    mask = S >= config.sig_rho_max - 0.0000001
    S[mask] = config.rho_max
    S[~mask] = config.rho_min

    RR = U @ np.diag(S) @ U.T
    atom0.set_params(E[:, i], RR, b[i])
    atoms.add_atom(atom0)
    coefs.add_coef(graph.λ[i])

sig = atoms.calc_sig(coefs)
f_sig = f(sig)
print(f_sig)
