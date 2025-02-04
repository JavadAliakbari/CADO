import json
import pickle as pkl

import numpy as np
from tqdm import tqdm

from Atom.atoms import Atoms
from Atom.coefs import Coefs
from Graph_Atom.graph_atom import GraphAtom
from Graph_Atom.graph_func import GraphFunc
from utils.utils import *
from utils.algorithm_utils_orig import *


def load_atom(path):
    atoms_path = f"{path}atoms.pkl"
    coefs_path = f"{path}coefs.pkl"
    graph_path = f"{path}graph.pkl"
    Lp_path = f"{path}Lp.pkl"
    Lambda_path = f"{path}Lambda.pkl"

    with open(atoms_path, "rb") as fid:
        atoms = pkl.load(fid)
    with open(coefs_path, "rb") as fid:
        coefs = pkl.load(fid)
    with open(graph_path, "rb") as fid:
        graph = pkl.load(fid)
    with open(Lp_path, "rb") as fid:
        Lp = pkl.load(fid)
    with open(Lambda_path, "rb") as fid:
        Lambda = pkl.load(fid)

    config.tau = graph.num_nodes

    return atoms, coefs, graph, Lp, Lambda


if __name__ == "__main__":
    path = "results/20250119_221156/"
    atoms, coefs, graph, Lp, Lambda = load_atom(path)
