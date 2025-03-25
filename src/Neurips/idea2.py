import numpy as np
from tqdm import tqdm

from Graph_Atom.graph_func import GraphFunc
from utils.algorithm_utils_orig import *
from utils.utils import *


def map_to_simplex(atoms: Atoms, coefs: Coefs, grads=None, g=None, lr=None):
    # s = np.array([np.sign(atom.e) for atom in atoms]).T
    Piv = np.array([coef * atom.e**2 for coef, atom in zip(coefs, atoms)]).T
    if grads is not None:
        grad_Piv = np.array(
            [
                2 * coef * grad_atom.e**2 + 2 * grad_c * atom.e**2
                for grad_c, grad_atom, coef, atom in zip(g, grads, coefs, atoms)
            ]
        ).T
        dd = np.mean(grad_Piv, axis=1)
        grad_Piv -= dd[:, None]
        Piv -= lr * grad_Piv
    Piv = project_to_simplex(Piv, 1)
    # Piv = Piv / np.sum(Piv, axis=1)[:, None]
    c = np.sum(Piv, axis=0)
    coefs.update_coefficients(c)
    E2 = Piv / (c[None, :] + 1e-10)
    E = np.sqrt(E2)
    # E = s * np.sqrt(E2)
    for atom_idx, atom in enumerate(atoms):
        atom.e = E[:, atom_idx]
        if grads is not None:
            atom.R -= 2 * coefs[atom_idx] * lr * grads[atom_idx].R
            atom.b -= 2 * coefs[atom_idx] * lr * grads[atom_idx].b

            atom.R = project_to_psd_cone(atom.R, config.rho_min, config.rho_max)
            atom.b = project_to_simplex(atom.b, 1)

    return atoms, coefs


def CANO(graph, atoms=None, coefs=None, log_results=True):
    n = graph.num_nodes
    num_features = graph.num_features
    c = graph.num_clusters
    ## Initialization
    if atoms is None:
        atoms, coefs = create_random_atoms(n, num_features, c, c)
    atoms_opt, coefs_opt = create_optimal_atoms(graph)
    num_atoms = len(atoms)
    # coefs = coefs_opt
    # atoms = atoms_opt

    atoms, coefs = map_to_simplex(atoms, coefs, None, None)
    sig = atoms.calc_sig(coefs)
    opt_sig = atoms_opt.calc_sig(coefs_opt)
    # sig = EmptySig(n, num_features, c)
    f = GraphFunc(graph=graph)
    f_opt = f(opt_sig)

    f_sig = f(sig)

    # if log_results:
    bar = tqdm(total=config.iters, leave=False)
    f_values = [f_sig]
    accuracies = []
    epsilsons = []
    components = []
    # coefs.update_coefficients(coefs_opt.get_coefs())
    for ii in range(config.iters):
        grads = []
        for ind in range(num_atoms):
            sig = atoms.calc_sig(coefs)  # Update signal
            grad = f.calc_grad(sig)
            grad_atom = atoms[ind].calc_grad(grad)
            grads.append(grad_atom)
        grads = np.array(grads)

        g = np.array(
            [grad.dot(atom.calc_sig_term()) for atom in atoms]
        )  # Gradient of the function

        atoms, coefs = map_to_simplex(atoms, coefs, grads, g, config.lr)

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
            # "beta": config.beta,
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
