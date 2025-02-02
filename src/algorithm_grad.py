import json
import pickle as pkl

import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

from src.utils.SBM import SBM
from src.Atom.coefs import Coefs
from src.Atom.atoms import Atoms
from src.Graph_Atom.graph_func import GraphFunc
from src.algorithm import Forward_step, Truncate, Enhancemnet_step, Gamma_line_search
from src.Graph_Atom.graph_sig import EmptySig
from src.utils.perfect_graph import PerfectGraph
from src.utils.algorithm_utils import *
from src.utils.utils import *

# Set print options to show the entire array
np.set_printoptions(threshold=np.inf, suppress=False)
# np.set_printoptions(suppress=True)


def Forward_step(atoms: Atoms, f: GraphFunc, coefs: Coefs, sig, itr):
    R = np.zeros((f.graph.num_features, f.graph.num_features))
    for i in range(10):
        grad = f.calc_grad(sig)
        new_atom = find_next_atom2(grad, R, itr)
        new_atoms = deepcopy(atoms)
        new_atoms.add_atom(new_atom)
        new_coefs = deepcopy(coefs)
        new_coefs.add_coef(config.tau)

        # Line 6 of CoGEnT
        if config.line_search:
            gamma = Gamma_line_search(atoms, f, coefs)
        else:
            gamma = 2 / (itr + 3)

        # Line 7 of CoGEnT(Line Search)
        new_coefs.scale_coefficients(factor=gamma)

        new_coefs = Enhancemnet_step(new_atoms, f, new_coefs)

        update_step(new_atoms, f, new_coefs)
        new_coefs = Enhancemnet_step(new_atoms, f, new_coefs)
        new_sig = new_atoms.calc_sig(new_coefs)
        sig.Rv = new_sig.Rv
        R = new_atoms[-1].R
        # grad = f.calc_grad(sig)

    coefs.add_coef(config.tau)
    atoms.add_atom(new_atom)

    # Line 6 of CoGEnT
    if config.line_search:
        gamma = Gamma_line_search(atoms, f, coefs)
    else:
        gamma = 2 / (itr + 3)

    # Line 7 of CoGEnT(Line Search)
    coefs.scale_coefficients(factor=gamma)

    return coefs


def update_step(atoms: Atoms, f: GraphFunc, coefs: Coefs):
    for i in range(100):
        sig = atoms.calc_sig(coefs)
        grad = f.calc_grad(sig)
        atom: GraphAtom = atoms[-1]
        grad_R = np.einsum("ijk,i->jk", grad.Rv, atom.e**2)
        Rp = atom.R - 0.01 * grad_R
        Rp = project_to_psd_cone(Rp, config.rho_min, config.rho_max)
        atom.R = Rp


if __name__ == "__main__":
    #### CoGEnT algorithm
    # samples_per_cluster = [4, 3, 2]  # Number of training samples in each cluster
    samples_per_cluster = [100, 100, 100]  # Number of training samples in each cluster
    num_features = 10  # Dimension of the features at each node
    num_signal_features = 2
    n = sum(samples_per_cluster)
    c = len(samples_per_cluster)
    probs = np.full((c, c), 0.0)
    np.fill_diagonal(probs, 0.1)
    # probs = np.eye(c)
    config.tau = n
    graph = SBM(
        samples_per_cluster,
        probs,
        num_features,
        num_signal_features,
        train_ratio=1,
        # train_ratio=0.4,
    )
    plot_adj(graph.A, name="A")

    ## Initialization
    atoms = Atoms()
    coefs = Coefs()
    f = GraphFunc(graph=graph)

    # atom0 = GraphAtom()
    # atom0.initialize(n, num_features, c)
    # atoms.add_atom(atom0)
    # coefs.update_coefficients([config.tau])
    # sig = atoms.calc_sig(coefs)
    sig = EmptySig(n, num_features, c)
    f_sig = f(sig)
    bar = tqdm(total=config.cogent_iters)
    f_values = []
    accuracies = []
    num_atoms = []
    for itr in range(config.cogent_iters):
        # config.eps *= itr / (itr + 1)
        coefs = Forward_step(atoms, f, coefs, sig, itr)

        coefs = Enhancemnet_step(atoms, f, coefs)

        sig = atoms.calc_sig(coefs)
        f_sig_new = f(sig)
        F_th = config.eta * f_sig + (1 - config.eta) * f_sig_new

        atoms, coefs = Truncate(atoms, f, coefs, F_th)
        sig = atoms.calc_sig(coefs)
        f_sig = f(sig)

        y_pred = np.argmax(sig.pi, axis=1)
        y_pred_test = y_pred[graph.test_mask]
        if y_pred_test.shape[0] > 0:
            acc = calc_accuracy(y_pred_test, graph.y[graph.test_mask])
        else:
            acc = 1

        bar.set_postfix(
            {
                "num atoms": len(atoms),
                "f": int(f_sig),
                "acc": acc,
            }
        )
        bar.update()
        f_values.append(f_sig)
        num_atoms.append(len(atoms))
        accuracies.append(acc)

    CoGEnT_sig = atoms.calc_sig(coefs)
    results = {
        f"atom {id}": {
            "e": str(np.round(atom.e, 2)),
            "b": str(np.round(atom.b, 2)),
            "R": str(np.round(np.linalg.eigvalsh(atom.R), 2)),
            "λ": str(np.round(c, 2)),
        }
        for id, (c, atom) in enumerate(zip(coefs, atoms))
    }

    mask = np.argmax(CoGEnT_sig.pi, axis=1) == graph.y
    # Replace True with ✔ and False with ✘
    marks = np.array(["✔" if value else "✘" for value in mask])[:, None]

    LOGGER.info(json.dumps(results, indent=4))
    LOGGER.info(np.hstack((np.round(CoGEnT_sig.pi, 2), marks)))

    plot_adj(CoGEnT_sig.L)

    y_pred = np.argmax(CoGEnT_sig.pi, axis=1)
    y_pred_test = y_pred[graph.test_mask]
    if y_pred_test.shape[0] > 0:
        acc = calc_accuracy(y_pred_test, graph.y[graph.test_mask])
    else:
        acc = 0

    plt.figure()
    plt.plot(f_values[1:])
    plt.yscale("log")
    plt.savefig(f"{save_path}loss.png")
    # plt.show()
    plt.figure()
    plt.plot(num_atoms)
    plt.savefig(f"{save_path}num_atoms.png")
    plt.figure()
    plt.plot(accuracies)
    plt.savefig(f"{save_path}accuracies.png")
    plt.title("Accuracy")

    filename = f"{save_path}atoms.pkl"
    with open(filename, "wb") as fid:
        pkl.dump(atoms, fid)

    filename = f"{save_path}coefs.pkl"
    with open(filename, "wb") as fid:
        pkl.dump(coefs, fid)
    filename = f"{save_path}graph.pkl"
    with open(filename, "wb") as fid:
        pkl.dump(graph, fid)

    # print(np.sum((CoGEnT_sig.L < 0) | (CoGEnT_sig.L > 1)))
    # print(np.round(CoGEnT_sig.L, 2))
    # LOGGER.info(a)
    LOGGER.info(acc)
    # print(f_sig3)

    # print(coefs.coefs)
