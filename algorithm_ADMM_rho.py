from copy import deepcopy
import json
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm

from SBM import SBM
from coefs import Coefs
from atoms import Atoms
from graph_func_ADMM2 import GraphFuncADMM

from graph_sig import EmptySig
from perfect_graph import PerfectGraph
from algorithm_utils import *
from utils import *
import pickle as pkl

# Set print options to show the entire array
np.set_printoptions(threshold=np.inf, suppress=False)
# np.set_printoptions(suppress=True)


def Gamma_line_search(atoms: Atoms, f: GraphFuncADMM, coefs: Coefs, Lambda, Lp):
    """
    Perform a line search to find the optimal gamma value that minimizes the function f.

    This function uses a binary search approach to find the gamma value in the range [0, 1]
    that minimizes the function f when applied to the signal calculated from the scaled coefficients.

    Parameters:
    atoms (Atoms): An object that contains the method `calc_sig` to calculate the signal from coefficients.
    f (GraphFunc): A function that takes a signal and returns a scalar value to be minimized.
    coefs (Coefs): An object that contains the method `scale_coefs` to scale the coefficients by a given gamma value.

    Returns:
    float: The gamma value in the range [0, 1] that minimizes the function f.
    """
    gamma_min, gamma_max = 0, 1

    for _ in range(10):
        gamma_mean = (gamma_max + gamma_min) / 2
        coefs_gamma_mean = Coefs.scale_coefs(coefs, gamma_mean)
        sig_gamma_mean = atoms.calc_sig(coefs_gamma_mean)
        f_gamma_mean = f(sig_gamma_mean, Lambda, Lp)

        if f_gamma_mean < f(
            atoms.calc_sig(Coefs.scale_coefs(coefs, gamma_min)), Lambda, Lp
        ):
            gamma_max = gamma_mean
        else:
            gamma_min = gamma_mean

    return gamma_min


def remove_atom(atoms: Atoms, f: GraphFuncADMM, coefs: Coefs, Lambda, Lp) -> int:
    """
    Identifies the index of the atom whose removal minimizes the value of the given function `f`
    when applied to the residual signature.

    Parameters:
    atoms (Atoms): The collection of atoms.
    f (GraphFunc): A function that evaluates the residual signature.
    coefs (Coefs): The coefficients associated with the atoms.

    Returns:
    int: The index of the atom whose removal minimizes the value of `f`. Returns 0 if there is only one coefficient.
    """
    if len(coefs) == 1:
        return 0
    # zxzjm
    # sig = atoms.calc_sig(coefs)

    min_val = np.inf
    arg_min_idx = -1
    for idx in range(len(coefs)):
        new_coefs = deepcopy(coefs)
        new_atoms = deepcopy(atoms)
        new_coefs.remove_coef(idx)
        # new_atoms.remove_atom(atoms[idx])
        new_atoms.remove_idx(idx)
        c_new = project_to_simplex(new_coefs, config.tau)
        new_coefs = Coefs(c_new)
        if config.enhance:
            new_coefs = Enhancemnet_step(new_atoms, f, new_coefs, Lambda, Lp)
        residual_sig = new_atoms.calc_sig(new_coefs)
        # single_sig = atoms[idx].calc_sig_term()
        # single_sig = coefs[idx] * single_sig
        # residual_sig = sig - single_sig
        val = f(residual_sig, Lambda, Lp)

        if val < min_val:
            min_val = val
            arg_min_idx = idx

    return arg_min_idx


def Forward_step(atoms: Atoms, f: GraphFuncADMM, coefs: Coefs, sig, Lambda, Lp, itr):
    """
    Perform a forward step in the CoGEnT algorithm.

    Parameters:
    atoms (Atoms): The current set of atoms.
    f (GraphFunc): The graph function used to calculate the gradient.
    coefs (Coefs): The coefficients associated with the atoms.
    sig: The signal or data used for gradient calculation.
    itr (int): The current iteration number.

    Returns:
    Coefs: The updated coefficients after the forward step.
    """
    grad = f.calc_grad(sig, Lambda, Lp)

    new_atom = find_next_atom(grad, itr)
    coefs.add_coef(config.tau)
    atoms.add_atom(new_atom)

    # Line 6 of CoGEnT
    if config.line_search:
        gamma = Gamma_line_search(atoms, f, coefs, Lambda, Lp)
    else:
        gamma = 2 / (itr + 3)

    # Line 7 of CoGEnT(Line Search)
    coefs.scale_coefficients(factor=gamma)

    return coefs


def Enhancemnet_step(atoms: Atoms, f: GraphFuncADMM, coefs: Coefs, Lambda, Lp):
    """
    Perform the enhancement step for the given coefficients.

    Parameters:
    atoms (Atoms): The atoms object containing the necessary methods for calculations.
    f (GraphFunc): The graph function object used to calculate gradients and function values.
    coefs (Coefs): The coefficients object to be updated.

    Returns:
    Coefs: The updated coefficients object.

    Description:
    This function performs an enhancement step to update the coefficients using gradient projection.
    It iteratively adjusts the coefficients to minimize the function value while ensuring they lie
    within a simplex defined by a given tau value. The function uses a gradient projection method
    with backtracking line search to find the optimal step size. The process stops when the
    convergence tolerance is met or the step size becomes too small.
    """
    if len(coefs) <= 1:
        coefs.update_coefficients([config.tau])
        return coefs

    alpha_min, alpha_red, c1 = 1e-6, 0.5, 1e-3
    sig, grad = atoms.calc_sig(coefs), f.calc_grad(atoms.calc_sig(coefs), Lambda, Lp)

    for _ in range(config.GP_iters):
        g = np.array([grad.dot(atom.calc_sig_term()) for atom in atoms])
        alpha, stop_flag = 1, False

        while alpha > alpha_min and not stop_flag:
            c_new = project_to_simplex(coefs.get_coefs() - alpha * g, config.tau)
            new_coefs = Coefs(c_new)
            tol_check = np.linalg.norm(coefs.get_coefs() - c_new)

            if alpha == 1 and tol_check <= config.GP_tol:
                stop_flag = True

            if f(atoms.calc_sig(new_coefs), Lambda, Lp) <= f(sig, Lambda, Lp) + c1 * (
                (c_new - coefs.get_coefs()).T @ g
            ):
                break

            alpha *= alpha_red

        if stop_flag or alpha < alpha_min:
            return coefs

        coefs = deepcopy(new_coefs)
        sig = atoms.calc_sig(coefs)
        grad = f.calc_grad(sig, Lambda, Lp)

    return coefs


def Enhancemnet_step2(atoms: Atoms, f: GraphFuncADMM, coefs: Coefs, Lambda, Lp):
    alpha = 0.01
    stop_flag = False
    iter = 0
    while not stop_flag and iter < config.GP_iters:
        sig = atoms.calc_sig(coefs)
        grad = f.calc_grad(sig, Lambda, Lp)
        g = np.array(
            [grad.dot(atom.calc_sig_term()) for atom in atoms]
        )  # Gradient of the function

        # Projection onto the simplex
        c = coefs.get_coefs()
        c_new = project_to_simplex(c - alpha * g, config.tau)
        # c_new = c - alpha * g
        coefs.update_coefficients(c_new)

        # Check convergence tolerance
        tol_check = np.linalg.norm(c - c_new)
        if tol_check <= config.GP_tol:
            stop_flag = True

        iter += 1

    # atoms.update_coefficients(coefs=c)
    return coefs


def Truncate(atoms: Atoms, f: GraphFuncADMM, coefs: Coefs, F_th, Lambda, Lp):
    """
    Truncates the given set of atoms and coefficients based on a threshold function value.

    Parameters:
    atoms (Atoms): The set of atoms to be truncated.
    f (GraphFunc): A function that evaluates the significance of the atoms.
    coefs (Coefs): The coefficients associated with the atoms.
    F_th (float): The threshold value for the function evaluation.

    Returns:
    tuple: A tuple containing the truncated set of atoms and the updated coefficients.
    """
    terminate = False
    while not terminate:
        remove_idx = remove_atom(atoms, f, coefs, Lambda, Lp)

        new_atoms = deepcopy(atoms)
        new_coefs = deepcopy(coefs)
        new_atoms.remove_atom(new_atoms[remove_idx])
        new_coefs.remove_coef(remove_idx)

        if len(new_coefs) > 0:
            new_c = project_to_simplex(new_coefs, config.tau)
            new_coefs.update_coefficients(new_c)
            new_coefs = Enhancemnet_step(new_atoms, f, new_coefs, Lambda, Lp)
            new_sig = new_atoms.calc_sig(new_coefs)
            f_new_sig = f(new_sig, Lambda, Lp)

            if f_new_sig <= F_th:
                atoms = new_atoms
                coefs = new_coefs
            else:
                terminate = True
        else:
            terminate = True

    return atoms, coefs


if __name__ == "__main__":
    #### CoGEnT algorithm
    # samples_per_cluster = [4, 3, 2]  # Number of training samples in each cluster
    samples_per_cluster = [100, 100, 100]  # Number of training samples in each cluster
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
    f = GraphFuncADMM(graph=graph)

    # atom0 = GraphAtom()
    # atom0.initialize(n, num_features, c)
    # atoms.add_atom(atom0)
    # coefs.update_coefficients([config.tau])
    # sig = atoms.calc_sig(coefs)

    Lambda = np.zeros((n, c))
    sig = EmptySig(n, num_features, c)

    Lp = sig.pi
    Lp = project_to_simplex2(Lp, 1)
    f_sig = f(sig, Lambda, Lp)
    bar = tqdm(total=config.cogent_iters)
    f_values = []
    accuracies = []
    diff_vals = []
    Lambda_terms = []
    num_atoms = []
    for ii in range(config.cogent_iters):
        for itr in range(config.local_iter):
            # config.eps *= itr / (itr + 1)
            coefs = Forward_step(atoms, f, coefs, sig, Lambda, Lp, itr)

            coefs = Enhancemnet_step(atoms, f, coefs, Lambda, Lp)

            sig = atoms.calc_sig(coefs)
            f_sig_new = f(sig, Lambda, Lp)
            F_th = config.eta * f_sig + (1 - config.eta) * f_sig_new

            atoms, coefs = Truncate(atoms, f, coefs, F_th, Lambda, Lp)
            sig = atoms.calc_sig(coefs)

        f_sig = f(sig, Lambda, Lp)

        # min Lp
        # CoGEnT_sig = atoms.calc_sig(coefs)
        # Lp += config.alpha * (Lambda + config.beta * (CoGEnT_sig.L - Lp))
        Lp = Lambda / config.beta + sig.pi
        Lp = project_to_simplex2(Lp, 1)

        # max Lambda
        Lambda += config.beta * (sig.pi - Lp)

        d = sig.pi - Lp
        diff = matrix_dot_product(d, d)

        lambda_term = matrix_dot_product(Lambda, d)

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
    plt.yscale("log")
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
