from copy import deepcopy
import numpy as np
from tqdm import tqdm
from scipy.special import softmax

from Graph_Atom.graph_func import GraphFunc
from Graph_Atom.graph_sig import GraphSig
from utils.algorithm_utils import *
from utils.utils import *


def clac_grad_ei(grad_sig: GraphSig, atom: Atoms, e):
    if grad_sig.u_v is not None:
        tmp1 = np.einsum("ij,j->i", grad_sig.u_v, atom.theta)
    else:
        tmp1 = 0
    tmp2 = np.einsum("ij,j->i", grad_sig.pi, atom.b)

    grad_ei_1 = (tmp1 + tmp2) * e
    grad_ei_2 = grad_sig.L @ e
    # grad_ei_2 = np.einsum("ij, j-> i", grad_sig.L, self.e)
    grad_ei = grad_ei_1 + grad_ei_2
    return grad_ei


def norm1_update(grad, atoms: Atoms, ii: int = 0, soft_assignment=True):
    # coefs = [sum(atom.e**2) for atom in atoms]
    # E = np.array([atom.e / np.sqrt(c) for atom, c in zip(atoms, coefs)])
    # grad_Piv = np.array(
    #     [
    #         coef * clac_grad_ei(grad, atom, e) ** 1
    #         for atom, e, coef in zip(atoms, E, coefs)
    #     ]
    # ).T
    grad_Piv = np.array([(atom.calc_grad_ei(grad) ** 1) for atom in atoms]).T

    if soft_assignment:
        grad_norm = np.linalg.norm(grad_Piv)
        s = algorithm.soft_assign_s
        temperature = max(1e-5, (1 - s * ii / algorithm.iters)) * grad_norm
        new_Piv = softmax(-grad_Piv / temperature, axis=1)
        new_Piv = project_to_simplex(new_Piv, 1)
    else:
        # Convert argmin to one-hot matrix
        min_indices = np.argmin(grad_Piv, axis=1)
        new_Piv = np.eye(grad_Piv.shape[1])[min_indices]
    new_E = np.sqrt(new_Piv)

    return new_E


def update(
    atoms: Atoms,
    f: GraphFunc,
    ii: int = 0,
    soft_assignment=True,
    norm2=False,
):
    sig = atoms.calc_sig()
    grad = f.calc_grad(sig)

    if norm2:
        grad_Piv = np.array([(atom.calc_grad_ei(grad) ** 1) for atom in atoms]).T
        new_E = -grad_Piv / np.linalg.norm(grad_Piv, axis=1, keepdims=True)
    else:
        new_E = norm1_update(grad, atoms, ii, soft_assignment)

    X_i = np.einsum("ik,ij->kj", new_E**2, f.graph.X)
    y_i = np.einsum(
        "ik,ij->kj", new_E[f.graph.train_nodes] ** 2, f.graph.pi[f.graph.train_nodes]
    )

    new_atoms = deepcopy(atoms)
    for idx in range(len(atoms)):
        new_atoms[idx].e = new_E[:, idx]
        # atoms[idx].update_theta(grad.u_v)
        # atoms[idx].update_b(grad.pi)
        new_atoms[idx].theta = project_to_ball(X_i[idx], data.tau)
        new_atoms[idx].b = np.zeros(f.graph.pi.shape[1], dtype=float)
        new_atoms[idx].b[np.argmax(y_i[idx])] = 1

    # alpha = 2 / (ii + 2)
    alpha = schedule_lr(ii)
    atoms = interpolate(atoms, new_atoms, alpha)
    # atoms = new_atoms

    # Sort atoms and coefs based on np.argmax(atom.b)
    sort_indices = np.argsort([np.argmax(atom.b) for atom in atoms])
    atoms = Atoms([atoms[i] for i in sort_indices])

    return atoms


def CADO(
    graph,
    atoms=None,
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
        atoms = create_random_atoms(n, num_features, c, c)
    atoms_opt = create_optimal_atoms(graph)

    atoms = map_to_simplex(atoms)
    sig = atoms.calc_sig()

    # Initialize GraphFunc with the current mu0 value
    f = GraphFunc(graph=graph, **kwargs)

    opt_sig = atoms_opt.calc_sig()
    f_opt, L_opt, R_opt, pi_opt = f(opt_sig, components=True)

    f_sig = f(sig)

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
        atoms = update(atoms, f, ii)

        sig = atoms.calc_sig()
        f_sig, L_term, R_term, pi_term = f(sig, components=True)

        if mu2 is None or mu2 == 0:
            E = np.array([atom.e for atom in atoms])
            y_pred = np.argmax(E, axis=0)
            acc = clustering_accuracy(y_pred, graph.y)
        else:
            y_pred = np.argmax(sig.pi, axis=1)
            y_pred_test = y_pred[graph.test_mask]
            if y_pred_test.shape[0] > 0:
                acc = calc_accuracy(y_pred_test, graph.y[graph.test_mask])
            else:
                acc = 1

        results = {
            f"f": f"{f_sig:,.4f}",
            "acc": f"{acc:,.4f}",
            "f_opt": f"{f_opt:,.4f}",
        }
        # y_pred = np.argmax(sig.pi, axis=1)
        # Store theta (k-means centers) instead of eigenvectors
        Q = np.array([atom.theta for atom in atoms])

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
            # print(f"Early stopping at iteration {ii} due to convergence")
            break

    return (
        atoms,
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
