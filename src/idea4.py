import numpy as np
from tqdm import tqdm

from Graph_Atom.graph_func import GraphFunc
from utils.algorithm_utils import *
from utils.utils import *


def line_search2(atoms1, coefs1, atoms2, coefs2, f):
    # Perform line search to find optimal step size
    # This is a placeholder implementation; you may need to adjust it
    alpha = 1.0
    beta = 0.5
    sig1 = atoms1.calc_sig(coefs1)
    f1 = f(sig1)
    for i in range(20):
        atoms = (1 - alpha) * atoms1 + alpha * atoms2
        coefs = (1 - alpha) * coefs1 + alpha * coefs2
        sig2 = atoms.calc_sig(coefs)
        f2 = f(sig2)
        if f2 < f1:
            break
        alpha *= beta
    return atoms, coefs


def update(atoms: Atoms, coefs: Coefs, f):
    sig = atoms.calc_sig(coefs)
    grad = f.calc_grad(sig)
    # Piv = np.array([coef * atom.e**2 for coef, atom in zip(coefs, atoms)]).T
    grad_Piv = np.array(
        [
            2 * coef * atom.calc_grad_ei(grad)
            + grad.dot(atom.calc_sig_term()) * (atom.e**2)
            for coef, atom in zip(coefs, atoms)
        ]
    ).T

    # Convert argmin to one-hot matrix
    min_indices = np.argmin(grad_Piv, axis=1)
    new_Piv = np.eye(grad_Piv.shape[1])[min_indices]
    new_E, new_c = Piv2EC(new_Piv)
    new_coefs = Coefs(new_c)
    new_atoms = Atoms()
    for idx, (atom, coef) in enumerate(zip(atoms, coefs)):
        new_atom = GraphAtom()
        new_atom.set_params(new_E[:, idx], atom.R, atom.b)
        new_sig = sig - atom.calc_sig_term()
        new_grad = f.calc_grad(new_sig)
        grad_Rv = 2 * coef * new_grad.Rv
        grad_pi = 2 * coef * new_grad.pi
        new_atom.update_R(grad_Rv)
        new_atom.update_b(grad_pi)
        new_atoms.add_atom(new_atom)

    atoms, coefs = line_search(atoms, coefs, new_atoms, new_coefs, f)

    return atoms, coefs


def CANO(graph, atoms=None, coefs=None, log_results=True):
    n = graph.num_nodes
    num_features = graph.num_features
    c = graph.num_clusters
    ## Initialization
    if atoms is None:
        atoms, coefs = create_random_atoms(n, num_features, c, c)
    atoms_opt, coefs_opt = create_optimal_atoms(graph)
    # num_atoms = len(atoms)
    # optimizer = get_optimizer(optimizer.type)
    # coefs = coefs_opt
    # atoms = atoms_opt

    atoms, coefs = map_to_simplex(atoms, coefs)
    sig = atoms.calc_sig(coefs)
    opt_sig = atoms_opt.calc_sig(coefs_opt)
    # sig = EmptySig(n, num_features, c)
    f = GraphFunc(graph=graph)
    f_opt = f(opt_sig)

    f_sig = f(sig)
    f_old = 0

    # if log_results:
    bar = tqdm(total=algorithm.iters, leave=False)
    f_values = [f_sig]
    accuracies = []
    epsilsons = []
    components = []
    # coefs.update_coefficients(coefs_opt.get_coefs())
    for ii in range(algorithm.iters):
        atoms, coefs = update(atoms, coefs, f)

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

        # if np.abs(f_sig - f_old) < 1e-6:
        #     break
        # f_old = f_sig

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
