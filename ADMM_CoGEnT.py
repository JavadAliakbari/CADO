import json
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm

from SBM import SBM
from algorithm import Enhancemnet_step, Forward_step, Truncate
from coefs import Coefs
from atoms import Atoms
from graph_func_ADMM import GraphFuncADMM
from graph_sig import EmptySig
from algorithm_utils import *
from utils import *
import pickle as pkl

# Set print options to show the entire array
np.set_printoptions(threshold=np.inf, suppress=False)
# np.set_printoptions(suppress=True)


if __name__ == "__main__":
    #### CoGEnT algorithm
    # samples_per_cluster = [4, 3, 2]  # Number of training samples in each cluster
    samples_per_cluster = [4, 3, 2]  # Number of training samples in each cluster
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
        # train_ratio=0.4,
    )
    plot_adj(graph.A, name="A")

    ## Initialization
    atoms = Atoms()
    coefs = Coefs()
    sig = EmptySig(n, num_features, c)

    Lambda = np.zeros((n, n))
    Lp = sig.L
    Lp = project_to_bound(Lp)

    f = GraphFuncADMM(graph=graph)
    f.set_ADMM_params(Lambda, Lp)

    bar = tqdm(total=config.cogent_iters)
    f_values = []
    accuracies = []
    diff_vals = []
    Lambda_terms = []
    num_atoms = []
    for ii in range(config.cogent_iters):
        for itr in range(config.local_iter):
            # sig = atoms.calc_sig(coefs)
            f_sig = f(sig)
            coefs = Forward_step(atoms, f, coefs, sig, ii * config.local_iter + itr)

            coefs = Enhancemnet_step(atoms, f, coefs)

            sig = atoms.calc_sig(coefs)
            f_sig_new = f(sig)
            F_th = config.eta * f_sig + (1 - config.eta) * f_sig_new

            atoms, coefs = Truncate(atoms, f, coefs, F_th)
            sig = atoms.calc_sig(coefs)

        # min Lp
        # Lp += config.alpha * (Lambda + config.beta * (CoGEnT_sig.L - Lp))
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

        bar.set_postfix(
            {
                "num atoms": len(atoms),
                "f": int(f_sig),
                "acc": acc,
                "diff": diff,
                "lambda_term": lambda_term,
            }
        )
        bar.update()
        f_values.append(f_sig)
        diff_vals.append(diff)
        Lambda_terms.append(lambda_term)
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
    plt.plot(diff_vals[1:])
    plt.plot(Lambda_terms[1:])
    # plt.yscale("log")
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
