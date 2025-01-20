import json
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm

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

if __name__ == "__main__":
    #### CoGEnT algorithm
    samples_per_cluster = [100, 100, 100]  # Number of training samples in each cluster
    num_features = 6  # Dimension of the features at each node
    num_signal_features = 2
    n = sum(samples_per_cluster)
    c = len(samples_per_cluster)
    probs = np.full((c, c), config.q)
    np.fill_diagonal(probs, config.p)
    # probs = np.eye(c)
    config.tau = n
    graph = SBM(
        samples_per_cluster,
        probs,
        num_features,
        num_signal_features,
        # train_ratio=1,
        train_ratio=config.train_ratio,
    )
    plot_adj(graph.A, name="A")

    ## Initialization
    atoms = Atoms()
    coefs = Coefs()

    E = graph.E
    # E[np.abs(E) < 0.00001] = 0.00001
    R = graph.Ri_matrices
    b = graph.b

    num_atoms = c
    for i in range(num_atoms):
        atom0 = GraphAtom()
        atom0.initialize(n, num_features, c)
        # S, U = np.linalg.eigh(R[i])
        # mask = S >= config.sig_rho_max - 0.0000001
        # S[mask] = config.rho_max - config.rho_min
        # S[~mask] = 0

        # RR = U @ np.diag(S) @ U.T
        # atom0.set_params(E[:, i], RR, b[i])
        atoms.add_atom(atom0)
    coefs = n * np.random.rand(num_atoms)
    coefs = project_to_simplex(coefs, n)
    coefs = Coefs(coefs)
    # coefs.add_coef(graph.λ[i])

    sig = atoms.calc_sig(coefs)
    # sig = EmptySig(n, num_features, c)
    f = GraphFunc(graph=graph)
    f_sig = f(sig)

    if config.ADMM:
        Lambda = np.zeros((n, n))
        Lp = sig.L
        Lp = project_to_bound(Lp)

        f.set_ADMM_params(Lambda, Lp)

    bar = tqdm(total=config.cogent_iters)
    f_values = []
    accuracies = []
    diff_vals = []
    Lambda_terms = []
    num_atoms = []
    for ii in range(config.cogent_iters):
        for itr in range(config.local_iter):
            for atom, coef in zip(atoms, coefs):
                sig2 = sig - coef * atom.calc_sig_term()
                grad = f.calc_grad(sig2)
                # grad_atom = atom.calc_grad(grad)
                # atom.update(grad_atom, 2 * coef * 0.1)  # Update atom
                atom.conditional_gradient_update(grad, ii * config.local_iter + itr)
                sig = atoms.calc_sig(coefs)  # Update signal
                f_sig = f(sig)
                f_values.append(f_sig)
            coefs = Enhancemnet_step(atoms, f, coefs)
            # atoms, coefs = update_coefs(atoms, f, coefs, sig, 0.2, 20)  # Update coefs

        if config.ADMM:
            # min Lp
            Lp = f.Lp
            Lambda = f.Lambda
            Lp = Lambda / config.beta + sig.L
            Lp = project_to_bound(Lp)

            # max Lambda
            d = sig.L - Lp
            Lambda += config.beta * d
            diff = matrix_dot_product(d, d)

            lambda_term = matrix_dot_product(Lambda, d)

            f.set_ADMM_params(Lambda, Lp)
        f_sig = f(sig)

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

        if ii % 100000 == 0:
            log_results(atoms, coefs, graph, results, save_path, terminal=False)
            plot_results(f_values, num_atoms, accuracies, atoms, save_path)

    log_results(atoms, coefs, graph, results, save_path)
    plot_results(f_values, num_atoms, accuracies, atoms, save_path)
    save_model_params(atoms, coefs, graph, Lambda, Lp, save_path)
