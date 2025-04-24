from copy import deepcopy

import numpy as np
from tqdm import tqdm
from scipy.special import softmax

from Graph_Atom.graph_func import GraphFunc
from utils.algorithm_utils import *
from utils.utils import *


def schedule_lr(
    ii,
    max_iter=algorithm.iters,
    min_lr=optimizer.min,
    max_lr=optimizer.lr,
    warmup_steps=algorithm.warmup_steps,
    decay_steps=optimizer.decay_steps,
    decay_rate=optimizer.decay_rate,
    method=optimizer.method,
    # method="exponential",
    # method="cosine",
):
    if method == "cosine":
        lr = min_lr + 0.5 * (max_lr - min_lr) * (
            1 + np.cos(np.pi * (ii - warmup_steps) / (max_iter - warmup_steps))
        )
        if ii < warmup_steps:
            lr = max_lr + (min_lr - max_lr) * (ii / warmup_steps)
    elif method == "linear":
        lr = max_lr + (min_lr - max_lr) * (ii / max_iter)
        if ii < warmup_steps:
            lr = max_lr + (min_lr - max_lr) * (ii / warmup_steps)
    elif method == "exponential":
        lr = max_lr + (min_lr - max_lr) * (ii / max_iter)
        if ii < warmup_steps:
            lr = max_lr + (min_lr - max_lr) * (ii / warmup_steps)
        elif ii < max_iter - decay_steps:
            lr = max_lr
        else:
            lr = max_lr + (min_lr - max_lr) * np.exp(
                -decay_rate * (ii - max_iter + decay_steps)
            )
    return lr


def update(
    atoms: Atoms,
    coefs: Coefs,
    f,
    ii,
    lr_search=True,
):
    sig = atoms.calc_sig(coefs)
    grad = f.calc_grad(sig)
    alpha = schedule_lr(ii)

    grad_coefs = np.array([grad.dot(atom.calc_sig_term()) for atom in atoms]).T
    min_indices = np.argmin(grad_coefs)
    new_c = Coefs((sig.n * np.eye(sig.c))[min_indices])
    new_coefs = (1 - alpha) * coefs + alpha * new_c
    E = np.array([atom.e for atom in atoms]).T
    new_E, new_coefs2 = find_eps(E, new_coefs.coefs)
    new_coefs = Coefs(new_coefs2)

    new_atoms = deepcopy(atoms)
    for idx in range(len(atoms)):
        new_atoms[idx].e = new_E[:, idx]
        new_sig = sig - new_atoms[idx].calc_sig_term()
        new_grad = f.calc_grad(new_sig, need_L=False)
        new_atoms[idx].update_R(new_grad.Rv)
        new_atoms[idx].update_b(new_grad.pi)

    if lr_search:
        atoms, coefs = line_search(atoms, coefs, new_atoms, new_coefs, f)
    else:
        atoms, coefs = interpolate(atoms, coefs, new_atoms, new_coefs, 1)

    # Sort atoms and coefs based on np.argmax(atom.b)
    sort_indices = np.argsort([np.argmax(atom.b) for atom in atoms])
    atoms = Atoms([atoms[i] for i in sort_indices])
    coefs = Coefs([coefs[i] for i in sort_indices])

    return atoms, coefs


def CANO(
    graph,
    atoms=None,
    coefs=None,
    mu0=None,
    mu1=None,
    lr_search=True,
    log_results=True,
    leave=True,
):
    n = graph.num_nodes
    num_features = graph.num_features
    c = graph.num_clusters
    ## Initialization
    if atoms is None:
        atoms, coefs = create_random_atoms(n, num_features, c, c)
    atoms_opt, coefs_opt = create_optimal_atoms(graph)

    # Initialize mu0 if not provided
    if mu0 is None:
        mu0 = data.mu0
    if mu1 is None:
        mu1 = data.mu1

    atoms, coefs = map_to_simplex(atoms, coefs)
    sig = atoms.calc_sig(coefs)
    # Initialize GraphFunc with the current mu0 value
    f = GraphFunc(graph=graph, mu0=mu0, mu1=mu1)

    opt_sig = atoms_opt.calc_sig(coefs_opt)
    f_opt, L_opt, R_opt, pi_opt = f(opt_sig, components=True)

    f_sig = f(sig)

    # if log_results:
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
        atoms, coefs = update(
            atoms,
            coefs,
            f,
            ii,
            lr_search=lr_search,
        )

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
            # "mu0": f"{mu0:,.4f}",  # Add mu0 to results
            # "mu1": f"{mu1:,.4f}",  # Add mu1 to results
        }
        y_pred = np.argmax(sig.pi, axis=1)
        Q = np.array([np.linalg.eigh(atom.R)[1] for atom in atoms])

        # if log_results:
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
