import numpy as np
from tqdm import tqdm

from Graph_Atom.graph_func import GraphFunc
from utils.Optimizer import get_optimizer
from utils.algorithm_utils import *
from utils.utils import *


def update(atoms: Atoms, coefs: Coefs, f, optimizer):
    sig = atoms.calc_sig(coefs)
    grad = f.calc_grad(sig)

    params = {}
    gradients = {}

    atom: GraphAtom
    for idx, atom in enumerate(atoms):
        params[f"R{idx}"] = atom.R
        params[f"b{idx}"] = atom.b
        gradients[f"R{idx}"] = 2 * coefs[idx] * atom.calc_grad_Ri(grad.Rv)
        gradients[f"b{idx}"] = 2 * coefs[idx] * atom.calc_grad_bi(grad.pi)

    # Use optimizer to update parameters
    updated_params = optimizer.step(params, gradients)

    for idx, atom in enumerate(atoms):
        atom.R = updated_params[f"R{idx}"]
        atom.b = updated_params[f"b{idx}"]

        atom.R = project_to_psd_cone(atom.R, data.rho_min, data.rho_max)
        atom.b = project_to_simplex(atom.b, 1)

    # sig = atoms.calc_sig(coefs)
    # grad = f.calc_grad(sig)
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
    alpha = optimizer.lr
    new_E, new_c = Piv2EC(new_Piv)
    new_coefs = Coefs(new_c)
    new_atoms = Atoms()
    for idx, atom in enumerate(atoms):
        new_atom = GraphAtom()
        new_atom.set_params(new_E[:, idx], atom.R, atom.b)
        new_atoms.add_atom(new_atom)

    atoms, coefs = interpolate(atoms, coefs, new_atoms, new_coefs, alpha)
    # sig = atoms.calc_sig(coefs)
    # grad = f.calc_grad(sig)

    return atoms, coefs


def CANO(graph, atoms=None, coefs=None, log_results=True):
    n = graph.num_nodes
    num_features = graph.num_features
    c = graph.num_clusters
    ## Initialization
    if atoms is None:
        atoms, coefs = create_random_atoms(n, num_features, c, c)
    atoms_opt, coefs_opt = create_optimal_atoms(graph)

    atoms, coefs = map_to_simplex(atoms, coefs)
    sig = atoms.calc_sig(coefs)
    opt_sig = atoms_opt.calc_sig(coefs_opt)
    f = GraphFunc(graph=graph)
    f_opt = f(opt_sig)

    f_sig = f(sig)
    optimizer = get_optimizer(optimizer.type)

    # if log_results:
    bar = tqdm(total=algorithm.iters, leave=False)
    f_old = 0
    f_values = [f_sig]
    accuracies = []
    epsilsons = []
    components = []

    for ii in range(algorithm.iters):
        atoms, coefs = update(atoms, coefs, f, optimizer)

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

        if np.abs(f_sig - f_old) < 1e-6:
            break
        f_old = f_sig

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
