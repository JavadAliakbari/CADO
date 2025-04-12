import pickle as pkl

from utils.utils import *
from utils.algorithm_utils_orig import *


def load_atom(path):
    atoms_path = f"{path}atoms.pkl"
    coefs_path = f"{path}coefs.pkl"
    graph_path = f"{path}graph.pkl"

    with open(atoms_path, "rb") as fid:
        atoms = pkl.load(fid)
    with open(coefs_path, "rb") as fid:
        coefs = pkl.load(fid)
    with open(graph_path, "rb") as fid:
        graph = pkl.load(fid)

    config.tau = graph.num_nodes

    return atoms, coefs, graph

if __name__ == "__main__":
    path = "results/20250119_221156/"
    atoms, coefs, graph, Lp, Lambda = load_atom(path)
