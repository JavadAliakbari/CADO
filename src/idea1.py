import numpy as np
from tqdm import tqdm

from Graph_Atom.graph_func import GraphFunc
from utils.algorithm_utils import *
from utils.utils import *


def CANO(graph, atoms=None, coefs=None, log_results=True):
    n = graph.num_nodes
    num_features = graph.num_features
    c = graph.num_clusters
    ## Initialization
    if atoms is None:
        atoms, coefs = create_random_atoms(n, num_features, c, c)
    atoms_opt, coefs_opt = create_optimal_atoms(graph)
    num_atoms = len(atoms)

    atoms, coefs = map_to_simplex(atoms, coefs)
    sig = atoms.calc_sig(coefs)
    opt_sig = atoms_opt.calc_sig(coefs_opt)
    f = GraphFunc(graph=graph)
    f_opt = f(opt_sig)

    f_sig = f(sig)

    # if log_results:
    bar = tqdm(total=algorithm.iters, leave=False)
    f_values = [f_sig]
    accuracies = []
    epsilsons = []
    components = []
    # coefs.update_coefficients(coefs_opt.get_coefs())
    for ii in range(algorithm.iters):
        idx = np.arange(num_atoms)
        np.random.shuffle(idx)

        for ind in idx:
            sig = atoms.calc_sig(coefs)  # Update signal
            grad = f.calc_grad(sig)
            grad_atom = 2 * coefs[ind] * atoms[ind].calc_grad(grad)
            atoms[ind].update(grad_atom)  # Update atom

            if ii % algorithm.local_iter == 0:
                # coefs = Enhancemnet_step2(atoms, f, coefs, iters=10, lr=0.0001)
                atoms, coefs = map_to_simplex(atoms, coefs)

        sig = atoms.calc_sig(coefs)
        f_sig, L_term, R_term, pi_term = f(sig, components=True)

        y_pred = np.argmax(sig.pi, axis=1)
        y_pred_test = y_pred[graph.test_mask]
        if y_pred_test.shape[0] > 0:
            acc = calc_accuracy(y_pred_test, graph.y[graph.test_mask])
        else:
            acc = 1

        results = {
            f"f": f"{f_sig:,.4f}",
            "acc": acc,
            "f_opt": f"{f_opt:,.4f}",
        }

        # if log_results:
        bar.set_postfix(results)
        bar.update()
        f_values.append(f_sig)
        accuracies.append(acc)
        epsilsons.append([atoms.e.copy() for atoms in atoms])
        components.append([L_term, R_term, pi_term])

    atoms, coefs = map_to_simplex(atoms, coefs)

    return (
        atoms,
        coefs,
        graph,
        results,
        f_values,
        accuracies,
        f_opt,
        epsilsons,
        components,
    )
