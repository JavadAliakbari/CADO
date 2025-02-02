import json
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm

from Atom.coefs import Coefs
from Graph_Atom.graph_atom import GraphAtom
from Atom.atoms import Atoms
from Graph_Atom.graph_func import GraphFunc

from utils.perfect_graph import PerfectGraph
from algorithm import Forward_step, Truncate, Enhancemnet_step
from utils.algorithm_utils import *
from utils.utils import *


if __name__ == "__main__":
    #### CoGEnT algorithm
    # samples_per_cluster = [4, 3, 2]  # Number of training samples in each cluster
    samples_per_cluster = [200, 150, 200]  # Number of training samples in each cluster
    num_features = 6  # Dimension of the features at each node
    num_signal_features = 2
    n = sum(samples_per_cluster)
    c = len(samples_per_cluster)
    config.tau = n
    graph = PerfectGraph(
        samples_per_cluster,
        num_features,
        num_signal_features,
        train_ratio=1,
        # train_ratio=0.4,
    )

    ## Initialization
    atoms = Atoms()
    coefs = Coefs()

    E = graph.E
    # E[np.abs(E) < 0.00001] = 0.00001
    R = graph.Ri_matrices
    b = graph.b

    for i in range(c - 3):
        atom0 = GraphAtom()
        S, U = np.linalg.eigh(R[i])
        mask = S >= config.sig_rho_max - 0.0000001
        S[mask] = config.rho_max
        S[~mask] = config.rho_min

        RR = U @ np.diag(S) @ U.T
        atom0.set_params(E[:, i], RR, b[i])
        atoms.add_atom(atom0)
        coefs.add_coef(graph.eigenvalues[i])

    f = GraphFunc(graph=graph)
    sig = atoms.calc_sig(coefs)
    f_sig = f(sig)
    bar = tqdm(total=config.cogent_iters)
    f_values = []
    accuracies = []
    num_atoms = []

    for itr in range(config.cogent_iters):
        # sig = atoms.calc_sig(coefs)
        # f_sig = f(sig)
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
                "f": f_sig,
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
    # a += [ for atom in atoms]
    # results["L"] = np.round(CoGEnT_sig.L, 2).tolist()
    # results["pi"] = np.round(CoGEnT_sig.pi, 2).tolist()
    # with open(f"{save_path}file.txt", mode="w") as fid:
    #     fid.write("\n".join(str(year) for year in a))

    LOGGER.info(json.dumps(results, indent=4))
    LOGGER.info(np.round(CoGEnT_sig.pi, 2))

    # Display the array as an image
    plt.figure()
    plt.imshow(CoGEnT_sig.L, cmap=custom_cmap)

    # Remove axis for a clean output
    plt.axis("off")

    # Save the image with a specific DPI (dots per inch) to control the output size
    plt.savefig(f"{save_path}L.png", dpi=300, bbox_inches="tight", pad_inches=0)

    y_pred = np.argmax(CoGEnT_sig.pi, axis=1)
    y_pred_test = y_pred[graph.test_mask]
    if y_pred_test.shape[0] > 0:
        acc = calc_accuracy(y_pred_test, graph.y[graph.test_mask])
    else:
        acc = 0

    plt.figure()
    plt.plot(f_values[1:])
    plt.savefig(f"{save_path}loss.png")
    # plt.show()
    plt.figure()
    plt.plot(num_atoms)
    plt.savefig(f"{save_path}num_atoms.png")
    plt.figure()
    plt.plot(accuracies)
    plt.savefig(f"{save_path}accuracies.png")
    plt.title("Accuracy")

    # print(np.sum((CoGEnT_sig.L < 0) | (CoGEnT_sig.L > 1)))
    # print(np.round(CoGEnT_sig.L, 2))
    # LOGGER.info(a)
    LOGGER.info(acc)
    # print(f_sig3)

    # print(coefs.coefs)
