from copy import deepcopy
import numpy as np
from tqdm import tqdm

from coefs import Coefs
from graph import Graph
from graph_atom import GraphAtom
from atoms import Atoms
from graph_func import GraphFunc

# from graph_grad import GraphGrad
from graph_sig import GraphSig
from perfect_graph import PerfectGraph
from algorithm_utils import *
from utils import *


def remove_atom(atoms: Atoms, f: GraphFunc, coefs: Coefs):
    sig = atoms.calc_sig(coefs)

    min_val = np.Infinity
    arg_min_idx = -1
    for idx in range(len(coefs)):
        single_sig = atoms[idx].calc_sig_term()
        single_sig = coefs[idx] * single_sig
        residual_sig = sig - single_sig
        val = f(residual_sig)

        if val < min_val:
            min_val = val
            arg_min_idx = idx

    return arg_min_idx


def Forward_step(atoms: Atoms, f: GraphFunc, coefs: Coefs):
    sig = atoms.calc_sig(coefs)
    # f.update_sig(sig)
    grad = f.calc_grad(sig)

    new_atom = find_next_atom(grad)
    coefs.add_coef(config.tau)
    atoms.add_atom(new_atom)
    # Dict.append(create_atom(eps, R, b))

    # Line 6 of CoGEnT
    gamma = 2 / (itr + 3)

    # Line 7 of CoGEnT(Line Search)
    coefs.scale_coefficients(factor=gamma)

    return coefs


def Enhancemnet_step(atoms: Atoms, f: GraphFunc, coefs: Coefs):
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
            print(" *** No suitable step length found ***")
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
        # grad.L *= -1
        grad.Rv *= config.mu1
        grad.pi *= -config.mu2
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
    samples_per_cluster = [4, 3, 2]  # Number of training samples in each cluster
    num_features = 5  # Dimension of the features at each node
    n = sum(samples_per_cluster)
    c = len(samples_per_cluster)
    config.tau = n
    graph = PerfectGraph(samples_per_cluster, num_features)

    ## Initialization
    atom0 = GraphAtom()
    atom0.initialize(n, num_features, c)
    coefs = Coefs()
    coefs.update_coefficients([config.tau])

    atoms = Atoms()
    atoms.add_atom(atom0)
    f = GraphFunc(graph=graph)

    iter = 0
    bar = tqdm(total=config.cogent_iters)
    for itr in range(config.cogent_iters):
        sig = atoms.calc_sig(coefs)
        f_sig = f(sig)
        coefs = Forward_step(atoms, f, coefs)

        coefs = Enhancemnet_step(atoms, f, coefs)

        sig = atoms.calc_sig(coefs)
        f_sig_new = f(sig)
        F_th = config.eta * f_sig + (1 - config.eta) * f_sig_new

        atoms, coefs = Truncate(atoms, f, coefs, F_th)
        bar.set_postfix(
            {
                "f": f_sig,
            }
        )
        bar.update()

a = [str(np.round(atom.e, 2)) + str(np.round(atom.b, 2)) for atom in atoms]
with open("./file.txt", mode="w") as fid:
    fid.write("\n".join(str(year) for year in a))

print(coefs.coefs)
