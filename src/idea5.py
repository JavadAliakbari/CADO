from copy import deepcopy

import numpy as np
from tqdm import tqdm
from scipy.special import softmax

from Graph_Atom.graph_func import GraphFunc
from utils.algorithm_utils import *
from utils.utils import *


def update(
    atoms: Atoms,
    coefs: Coefs,
    f,
    ii,
    prev_grad=None,
    momentum_beta=0.9,
    soft_assignment=True,
    lr_search=True,
):
    # ss = np.array([np.sign(atom.e) for atom in atoms]).T
    sig = atoms.calc_sig(coefs)
    grad = f.calc_grad(sig)

    # Continue with normal atom and coef updates
    # grad_Piv = []
    # for coef, atom in zip(coefs, atoms):
    #     atom_sig = atom.calc_sig_term()
    #     new_sig = sig - atom_sig
    #     new_grad = f.calc_grad(new_sig)
    #     grad_c = np.abs(new_grad @ atom_sig)
    #     grad_e = grad_c * (atom.calc_grad_ei(new_grad) ** 1)
    #     grad_Piv.append(grad_e)

    # grad_Piv = np.array(grad_Piv).T

    # grad_Piv = np.array(
    #     [coef * (atom.calc_grad_ei(grad) ** 1) for coef, atom in zip(coefs, atoms)]
    # ).T
    grad_coefs = np.abs([grad @ atom.calc_sig_term() for atom in atoms]).T
    grad_Piv = np.array(
        [
            grad_coef * (atom.calc_grad_ei(grad) ** 1)
            for grad_coef, atom in zip(grad_coefs, atoms)
        ]
    ).T
    # Apply momentum to gradient if we have previous gradient
    if prev_grad is not None:
        grad_Piv = momentum_beta * prev_grad + (1 - momentum_beta) * grad_Piv
    grad_norm = np.linalg.norm(grad_Piv)

    if soft_assignment:
        s = algorithm.soft_assign_s
        temperature = max(1e-5, (1 - s * ii / algorithm.iters)) * grad_norm
        new_Piv = softmax(-grad_Piv / temperature, axis=1)
        new_Piv = project_to_simplex(new_Piv, 1)
    else:
        # Convert argmin to one-hot matrix
        min_indices = np.argmin(grad_Piv, axis=1)
        new_Piv = np.eye(grad_Piv.shape[1])[min_indices]
    new_E, new_c = Piv2EC(new_Piv)

    new_coefs = Coefs(new_c)
    new_atoms = deepcopy(atoms)
    for idx in range(len(atoms)):
        new_atoms[idx].e = new_E[:, idx]
        # new_sig = sig - new_atoms[idx].calc_sig_term()
        # new_grad = f.calc_grad(new_sig, need_L=False)
        new_atoms[idx].update_R(grad.Rv)
        new_atoms[idx].update_b(grad.pi)

    if lr_search:
        alpha = schedule_lr(ii)
        atoms, coefs = line_search(atoms, coefs, new_atoms, new_coefs, f, alpha)
    else:
        alpha = schedule_lr(ii)
        atoms, coefs = interpolate(atoms, coefs, new_atoms, new_coefs, alpha)

    # Sort atoms and coefs based on np.argmax(atom.b)
    sort_indices = np.argsort([np.argmax(atom.b) for atom in atoms])
    atoms = Atoms([atoms[i] for i in sort_indices])
    coefs = Coefs([coefs[i] for i in sort_indices])

    return atoms, coefs, grad_Piv


def CANO(
    graph,
    atoms=None,
    coefs=None,
    lr_search=True,
    log_results=True,
    leave=True,
    **kwargs,
):
    n = graph.num_nodes
    num_features = graph.num_features
    c = graph.num_clusters

    mu2 = kwargs.get("mu2", data.mu2)
    ## Initialization
    if atoms is None:
        atoms, coefs = create_random_atoms(n, num_features, c, c)
    atoms_opt, coefs_opt = create_optimal_atoms(graph)

    atoms, coefs = map_to_simplex(atoms, coefs)
    sig = atoms.calc_sig(coefs)

    # Initialize GraphFunc with the current mu0 value
    f = GraphFunc(graph=graph, **kwargs)

    opt_sig = atoms_opt.calc_sig(coefs_opt)
    f_opt, L_opt, R_opt, pi_opt = f(opt_sig, components=True)

    f_sig = f(sig)
    prev_grad = None

    if log_results:
        bar = tqdm(total=algorithm.iters, leave=leave)
    f_values = [f_sig]
    accuracies = []
    epsilsons = []
    components = []
    y_preds = []
    Qs = []

    save_iters = np.linspace(
        0, algorithm.iters - 1, num=min(200, algorithm.iters), dtype=int
    ).tolist()

    for ii in range(algorithm.iters):
        atoms, coefs, current_grad = update(
            atoms,
            coefs,
            f,
            ii,
            prev_grad,
            lr_search=lr_search,
        )
        prev_grad = current_grad

        sig = atoms.calc_sig(coefs)
        f_sig, L_term, R_term, pi_term = f(sig, components=True)

        y_pred = np.argmax(sig.pi, axis=1)
        y_pred_test = y_pred[graph.test_mask]
        if y_pred_test.shape[0] > 0:
            if mu2 is None or mu2 == 0:
                acc = clustring_accuracy(y_pred_test, graph.y[graph.test_mask])
            else:
                acc = calc_accuracy(y_pred_test, graph.y[graph.test_mask])
        else:
            acc = 1

        results = {
            f"f": f"{f_sig:,.4f}",
            "acc": f"{acc:,.4f}",
            "f_opt": f"{f_opt:,.4f}",
        }
        y_pred = np.argmax(sig.pi, axis=1)
        Q = np.array([np.linalg.eigh(atom.R)[1] for atom in atoms])

        if log_results:
            bar.set_postfix(results)
            bar.update()
        f_values.append(f_sig)
        accuracies.append(acc)
        epsilsons.append([atoms.e.copy() for atoms in atoms])
        components.append([L_term, R_term, pi_term])

        # Change saving frequency using adaptive iterations
        if ii in save_iters:
            y_preds.append(y_pred)
            Qs.append(Q)

        # early stopping condition
        if ii > 100 and np.mean(np.abs(np.diff(f_values[-20:]))) < 1e-6:
            print(f"Early stopping at iteration {ii} due to convergence")
            break

    return (
        atoms,
        coefs,
        graph,
        results,
        f_values,
        accuracies,
        epsilsons,
        components,
        (f_opt, L_opt, R_opt, pi_opt),
        (y_preds, Qs),
        # mu0_values,  # Return the mu0 values history
    )
