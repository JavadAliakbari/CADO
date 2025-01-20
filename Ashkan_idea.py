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
    probs = np.full((c, c), 0.05)
    np.fill_diagonal(probs, 0.4)
    # probs = np.eye(c)
    config.tau = n
    graph = SBM(
        samples_per_cluster,
        probs,
        num_features,
        num_signal_features,
        # train_ratio=1,
        train_ratio=0.1,
    )
    plot_adj(graph.A, name="A")

    ## Initialization
    atoms = Atoms()
    coefs = Coefs()

    E = graph.E
    # E[np.abs(E) < 0.00001] = 0.00001
    R = graph.Ri_matrices
    b = graph.b

    num_atoms = c + 1
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
                grad = f.calc_grad(sig)
                grad_atom = atom.calc_grad(grad)
                atom.update(grad_atom, 2 * coef * 0.1)  # Update atom
                sig = atoms.calc_sig(coefs)  # Update signal
                f_sig = f(sig)
                f_values.append(f_sig)
            atoms, coefs = update_coefs(atoms, f, coefs, sig, 0.2, 20)  # Update coefs

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
    plt.plot(f_values)
    # plt.yscale("log")
    plt.savefig(f"{save_path}loss.png")
    plt.figure()
    plt.plot(num_atoms)
    plt.savefig(f"{save_path}num_atoms.png")
    plt.figure()
    plt.plot(accuracies)
    plt.title("Accuracy")
    plt.savefig(f"{save_path}accuracies.png")

    plt.figure()
    for id, atom in enumerate(atoms):
        plt.plot(atom.e, "*-", label=f"atom {id}")
    plt.legend()
    plt.title("Eigenvectors")
    plt.savefig(f"{save_path}eigenvectors.png")

    filename = f"{save_path}atoms.pkl"
    with open(filename, "wb") as fid:
        pkl.dump(atoms, fid)

    filename = f"{save_path}coefs.pkl"
    with open(filename, "wb") as fid:
        pkl.dump(coefs, fid)
    filename = f"{save_path}graph.pkl"
    with open(filename, "wb") as fid:
        pkl.dump(graph, fid)

    if config.ADMM:
        filename = f"{save_path}Lambda.pkl"
        with open(filename, "wb") as fid:
            pkl.dump(Lambda, fid)

        filename = f"{save_path}Lp.pkl"
        with open(filename, "wb") as fid:
            pkl.dump(Lp, fid)

    LOGGER.info(acc)
