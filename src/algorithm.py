import json
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm

from Graph_Atom.ADMM import ADMM_cls
from utils.SBM import SBM
from Atom.coefs import Coefs
from Atom.atoms import Atoms
from Graph_Atom.graph_func import GraphFunc

from Graph_Atom.graph_sig import EmptySig
from utils.algorithm_utils_orig import *
from utils.utils import *
import pickle as pkl

# Set print options to show the entire array
np.set_printoptions(threshold=np.inf, suppress=False)
# np.set_printoptions(suppress=True)


def check_constraint(atoms, coefs):
    sig = atoms.calc_sig(coefs)
    grad = f.calc_grad(sig)

    for _ in range(config.GP_iters):
        g = np.array([grad.dot(atom.calc_sig_term()) for atom in atoms])

    E = [atom.e for atom in atoms]
    E = np.array(E).T
    n, c = E.shape

    # l = n * np.random.rand(c)
    # l = n * np.ones(c) / c
    l = coefs.get_coefs()
    # l = project_to_simplex3(l, n)

    H = E**2 @ np.diag(l)
    A = project_inside_simplex(H, 1)
    new_l = np.sum(A, axis=0)

    new_E = A @ np.diag(1 / (new_l + 1e-10))
    for idx, atom in enumerate(atoms):
        atom.e = np.sign(atom.e) * np.sqrt(new_E[:, idx])

    coefs.update_coefficients(new_l)

    return atoms, coefs


if __name__ == "__main__":
    #### CoGEnT algorithm
    # samples_per_cluster = [4, 3, 2]  # Number of training samples in each cluster
    samples_per_cluster = [100, 100, 100]  # Number of training samples in each cluster
    num_features = 6  # Dimension of the features at each node
    num_signal_features = 2
    n = sum(samples_per_cluster)
    c = len(samples_per_cluster)
    probs = np.full((c, c), 0.0)
    np.fill_diagonal(probs, 1)
    # probs = np.eye(c)
    config.tau = n
    graph = SBM(
        samples_per_cluster,
        probs,
        num_features,
        num_signal_features,
        train_ratio=1,
        # train_ratio=0.5,
    )
    plot_adj(graph.A, name="A")

    ## Initialization
    atoms = Atoms()
    coefs = Coefs()
    sig = EmptySig(n, num_features, c)
    f = GraphFunc(graph=graph)

    if config.ADMM == "L":
        admm = ADMM_cls(sig.L, project_to_bound)
        f.set_ADMM_params(admm)
    elif config.ADMM == "pi":
        admm = ADMM_cls(sig.pi, project_to_simplex)
        f.set_ADMM_params(admm)
    else:  # config.ADMM == "None"
        admm = None
    f_sig = f(sig, ADMM=True)

    bar = tqdm(total=config.iters)
    f_values = []
    accuracies = []
    diff_vals = []
    Lambda_terms = []
    Lambda_norms = []
    num_atoms = []
    for ii in range(config.iters):
        for itr in range(config.local_iter):
            atoms, coefs = Forward_step(
                atoms, f, coefs, sig, ii * config.local_iter + itr
            )
            # atoms, coefs = check_constraint(atoms, coefs)
            # if sum(coefs) >= n:
            #     atoms, coefs = Enhance_step(atoms, coefs)

            coefs = Enhancemnet_step(atoms, f, coefs, ADMM=True)

            sig = atoms.calc_sig(coefs)
            f_sig_new = f(sig, ADMM=True)
            F_th = config.eta * f_sig + (1 - config.eta) * f_sig_new

            atoms, coefs, f = Truncate(atoms, f, coefs, F_th)
            # if len(num_atoms) > 2:
            #     atoms, coefs = Matrix_Truncate(atoms, f, coefs, F_th)
            sig = atoms.calc_sig(coefs)

        if admm is not None:
            if config.ADMM == "L":
                diff, lambda_term, lambda_norm = admm.ADMM_update(sig.L)
            elif config.ADMM == "pi":
                diff, lambda_term, lambda_norm = admm.ADMM_update(sig.pi)
            diff_vals.append(diff)
            Lambda_terms.append(lambda_term)
            Lambda_norms.append(lambda_norm)
        f_sig = f(sig, ADMM=True)

        y_pred = np.argmax(sig.pi, axis=1)
        y_pred_test = y_pred[graph.test_mask]
        if y_pred_test.shape[0] > 0:
            acc = calc_accuracy(y_pred_test, graph.y[graph.test_mask])
        else:
            acc = 1

        results = {
            "num atoms": len(atoms),
            f"f": f"{f_sig:,.4f}",
            "acc": acc,
        }

        if config.ADMM:
            results["diff"] = diff
            results["lambda term"] = lambda_term

        bar.set_postfix(results)
        bar.update()
        f_values.append(f_sig)
        num_atoms.append(len(atoms))
        accuracies.append(acc)

        if config.ADMM:
            diff_vals.append(diff)
            Lambda_terms.append(lambda_term)

    log_results(atoms, coefs, graph, results, save_path)
    plot_results(
        f_values,
        num_atoms,
        accuracies,
        Lambda_norms,
        atoms,
        admm.Z,
        save_path,
        # f_opt,
    )
    save_model_params(atoms, coefs, graph, admm, save_path)
