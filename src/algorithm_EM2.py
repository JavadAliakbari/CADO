from copy import deepcopy
import json
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm

from Atom.coefs import Coefs
from Graph_Atom.graph_atom import GraphAtom
from Atom.atoms import Atoms
from Graph_Atom.graph_func import GraphFunc

from Graph_Atom.graph_sig import EmptySig
from utils.perfect_graph import PerfectGraph
from utils.algorithm_utils import *
from utils.utils import *


def Gamma_line_search(atoms: Atoms, f: GraphFunc, coefs: Coefs, Lambda, Lp):
    gamma_min = 0
    gamma_max = 1

    coefs_gamma_min = Coefs.scale_coefs(coefs, gamma_min)
    coefs_gamma_max = Coefs.scale_coefs(coefs, gamma_max)

    sig_gamma_min = atoms.calc_sig(coefs_gamma_min)
    sig_gamma_max = atoms.calc_sig(coefs_gamma_max)

    f_gamma_min = f(sig_gamma_min, Lambda, Lp)
    f_gamma_max = f(sig_gamma_max, Lambda, Lp)

    if f_gamma_max < f_gamma_min:
        gamma_temp = gamma_min
        coefs_gamma_temp = deepcopy(coefs_gamma_min)
        sig_gamma_temp = deepcopy(sig_gamma_min)
        f_gamma_temp = f_gamma_min

        gamma_min = gamma_max
        coefs_gamma_min = deepcopy(coefs_gamma_max)
        sig_gamma_min = deepcopy(sig_gamma_max)
        f_gamma_min = f_gamma_max

        gamma_max = gamma_temp
        coefs_gamma_max = deepcopy(coefs_gamma_temp)
        sig_gamma_max = deepcopy(sig_gamma_temp)
        f_gamma_max = f_gamma_temp

    for iter in range(10):
        gamma_mean = (gamma_max + gamma_min) / 2
        coefs_gamma_mean = Coefs.scale_coefs(coefs, gamma_mean)
        sig_gamma_mean = atoms.calc_sig(coefs_gamma_mean)
        f_gamma_mean = f(sig_gamma_mean, Lambda, Lp)

        if f_gamma_mean < f_gamma_min:
            gamma_temp = gamma_min
            coefs_gamma_temp = deepcopy(coefs_gamma_min)
            sig_gamma_temp = deepcopy(sig_gamma_min)
            f_gamma_temp = f_gamma_min

            gamma_max = gamma_min
            coefs_gamma_max = deepcopy(coefs_gamma_min)
            sig_gamma_max = deepcopy(sig_gamma_min)
            f_gamma_max = f_gamma_min

            gamma_min = gamma_mean
            coefs_gamma_min = deepcopy(coefs_gamma_mean)
            sig_gamma_min = deepcopy(sig_gamma_mean)
            f_gamma_min = f_gamma_mean
        else:
            gamma_max = gamma_mean
            coefs_gamma_max = deepcopy(coefs_gamma_mean)
            sig_gamma_max = deepcopy(sig_gamma_mean)
            f_gamma_max = f_gamma_mean

    return gamma_min


def remove_atom(atoms: Atoms, f: GraphFunc, coefs: Coefs):
    # sig = atoms.calc_sig(coefs)

    min_val = np.inf
    arg_min_idx = -1
    for idx in range(len(coefs)):
        new_coefs = deepcopy(coefs)
        new_atoms = deepcopy(atoms)
        new_coefs.remove_coef(idx)
        # new_atoms.remove_atom(atoms[idx])
        new_atoms.remove_idx(idx)
        if len(new_coefs) > 0:
            c_new = project_to_simplex(new_coefs, config.tau)
            new_coefs = Coefs()
            new_coefs.update_coefficients(c_new)
        new_coefs = Enhancemnet_step(new_atoms, f, new_coefs)
        residual_sig = new_atoms.calc_sig(new_coefs)
        # single_sig = atoms[idx].calc_sig_term()
        # single_sig = coefs[idx] * single_sig
        # residual_sig = sig - single_sig
        val = f(residual_sig)

        if val < min_val:
            min_val = val
            arg_min_idx = idx

    return arg_min_idx


def Em_step(atoms: Atoms, f: GraphFunc, coefs: Coefs):
    for i in range(1):
        # sig = atoms.calc_sig(coefs)
        cov = f.calc_cov()
        # grad = f.calc_grad(sig)

        # R = atoms[-1].R
        R = update_R2(cov, atoms[-1], i)
        atoms[-1].R = R

        # coefs = Enhancemnet_step(atoms, f, coefs)

    # atoms[-1].R = f.graph.Rv_matrices[2]
    return coefs


def Em_step2(atoms: Atoms, f: GraphFunc, coefs: Coefs):
    for i in range(10):
        sig = atoms.calc_sig(coefs)
        grad = f.calc_grad(sig)

        R = atoms[-1].R
        R = update_R(grad, R)
        atoms[-1].R = R

        # coefs = Enhancemnet_step(atoms, coefs)

    return coefs


def Forward_step(atoms: Atoms, f: GraphFunc, coefs: Coefs, sig, itr):
    R = np.zeros((f.graph.num_features, f.graph.num_features))
    for i in range(10):
        grad = f.calc_grad(sig)
        new_atom = find_next_atom2(grad, R, itr)
        new_atoms = deepcopy(atoms)
        new_atoms.add_atom(new_atom)
        new_coefs = deepcopy(coefs)
        new_coefs.add_coef(config.tau)

        # Line 6 of CoGEnT
        if config.line_search:
            gamma = Gamma_line_search(atoms, f, coefs)
        else:
            gamma = 2 / (itr + 3)

        # Line 7 of CoGEnT(Line Search)
        new_coefs.scale_coefficients(factor=gamma)

        new_coefs = Enhancemnet_step(new_atoms, f, new_coefs)

        new_coefs = Em_step(new_atoms, f, new_coefs)
        new_coefs = Enhancemnet_step(new_atoms, f, new_coefs)
        new_sig = new_atoms.calc_sig(new_coefs)
        sig.Rv = new_sig.Rv
        R = new_atoms[-1].R
        # grad = f.calc_grad(sig)

    coefs.add_coef(config.tau)
    atoms.add_atom(new_atom)

    # Line 6 of CoGEnT
    if config.line_search:
        gamma = Gamma_line_search(atoms, f, coefs)
    else:
        gamma = 2 / (itr + 3)

    # Line 7 of CoGEnT(Line Search)
    coefs.scale_coefficients(factor=gamma)

    return coefs


def Enhancemnet_step(atoms: Atoms, f: GraphFunc, coefs: Coefs):
    # return coefs
    # if len(atoms) == 2:
    #     vals = []
    #     for i in range(config.tau + 1):
    #         c_new = [i, config.tau - i]
    #         coefs = Coefs()
    #         coefs.update_coefficients(c_new)
    #         sig = atoms.calc_sig(coefs)
    #         f_val = f(sig)
    #         vals.append(f_val)
    #     plt.plot(vals)
    #     plt.savefig(f"{save_path}res.png")
    #     aaa = 2

    alpha_min = 1e-6
    alpha_red = 0.5
    c1 = 1e-3
    for iter in range(config.GP_iters):
        sig = atoms.calc_sig(coefs)
        grad = f.calc_grad(sig)
        g = np.array(
            [grad.dot(atom.calc_sig_term()) for atom in atoms]
        )  # Gradient of the function

        alpha = 1
        stop_flag = False
        f_val = f(sig)

        while (alpha > alpha_min) and (not stop_flag):
            # Projection onto the simplex
            c = coefs.get_coefs()
            c_new = project_to_simplex(c - alpha * g, config.tau)
            new_coefs = Coefs()
            new_coefs.update_coefficients(c_new)

            # Check convergence tolerance
            tol_check = np.linalg.norm(c - c_new)
            if (alpha == 1) and (iter > 1):
                if tol_check <= config.GP_tol:
                    stop_flag = True

            # Compute the new residual and function value

            new_sig = atoms.calc_sig(new_coefs)
            f_new = f(new_sig)

            # Check the Armijo condition (sufficient decrease condition)
            if f_new <= f_val + c1 * ((c_new - c).T @ g):
                break

            alpha *= alpha_red  # Reduce step size

        if stop_flag:
            # Successful exit
            return coefs

        if alpha < alpha_min:
            # Unsuccessful exit due to too small step length
            # print(" *** No suitable step length found ***")
            return coefs

        coefs = new_coefs

    # atoms.update_coefficients(coefs=c)
    return coefs


def Enhancemnet_step2(atoms: Atoms, f: GraphFunc, coefs: Coefs):
    alpha = 0.01
    stop_flag = False
    iter = 0
    while not stop_flag and iter < config.GP_iters:
        sig = atoms.calc_sig(coefs)
        grad = f.calc_grad(sig)
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


def Truncate(atoms: Atoms, f: GraphFunc, coefs: Coefs, F_th):
    terminate = False
    while not terminate:
        remove_idx = remove_atom(atoms, f, coefs)

        new_atoms = deepcopy(atoms)
        new_coefs = deepcopy(coefs)
        new_atoms.remove_atom(new_atoms[remove_idx])
        new_coefs.remove_coef(remove_idx)
        new_c = project_to_simplex(new_coefs, config.tau)
        new_coefs.update_coefficients(new_c)

        if len(new_coefs) > 0:
            new_coefs = Enhancemnet_step(new_atoms, f, new_coefs)
            new_sig = new_atoms.calc_sig(new_coefs)
            f_new_sig = f(new_sig)

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
    samples_per_cluster = [100, 90, 80]  # Number of training samples in each cluster
    num_features = 10  # Dimension of the features at each node
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
    E[np.abs(E) < 0.00001] = 0.00001
    R = graph.Ri_matrices
    b = graph.b

    for i in range(c - 1):
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
    bar = tqdm(total=config.iters)
    f_values = []
    accuracies = []
    num_atoms = []
    for itr in range(config.iters):
        # config.eps *= itr / (itr + 1)
        coefs = Forward_step(atoms, f, coefs, sig, itr)

        # coefs = Enhancemnet_step(atoms, f, coefs)

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
