from copy import deepcopy
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm

from coefs import Coefs
from graph_atom import GraphAtom
from atoms import Atoms
from graph_func_ADMM import GraphFuncADMM

from perfect_graph import PerfectGraph
from algorithm_utils import *
from utils import *


def remove_atom(atoms: Atoms, f: GraphFuncADMM, coefs: Coefs, Lambda, Lp):
    sig = atoms.calc_sig(coefs)

    min_val = np.Infinity
    arg_min_idx = -1
    for idx in range(len(coefs)):
        single_sig = atoms[idx].calc_sig_term()
        single_sig = coefs[idx] * single_sig
        residual_sig = sig - single_sig
        val = f(residual_sig, Lambda, Lp)

        if val < min_val:
            min_val = val
            arg_min_idx = idx

    return arg_min_idx


def Forward_step(atoms: Atoms, f: GraphFuncADMM, coefs: Coefs, Lambda, Lp):
    sig = atoms.calc_sig(coefs)
    # f.update_sig(sig)
    grad = f.calc_grad(sig, Lambda, Lp)

    new_atom = find_next_atom(grad)
    coefs.add_coef(config.tau)
    atoms.add_atom(new_atom)
    # Dict.append(create_atom(eps, R, b))

    # Line 6 of CoGEnT
    gamma = 2 / (itr + 3)

    # Line 7 of CoGEnT(Line Search)
    coefs.scale_coefficients(factor=gamma)

    return coefs


def Enhancemnet_step(atoms: Atoms, f: GraphFuncADMM, coefs: Coefs, Lambda, Lp):
    alpha_min = 1e-6
    alpha_red = 0.5
    c1 = 1e-3
    for iter in range(config.GP_iters):
        sig = atoms.calc_sig(coefs)
        grad = f.calc_grad(sig, Lambda, Lp)
        g = np.array(
            [grad.dot(atom.calc_sig_term()) for atom in atoms]
        )  # Gradient of the function

        alpha = 1
        stop_flag = False
        f_val = f(sig, Lambda, Lp)

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
            f_new = f(new_sig, Lambda, Lp)

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


def Enhancemnet_step2(atoms: Atoms, f: GraphFuncADMM, coefs: Coefs, Lambda, Lp):
    alpha = 0.01
    stop_flag = False
    iter = 0
    while not stop_flag and iter < config.GP_iters:
        sig = atoms.calc_sig(coefs)
        grad = f.calc_grad(sig, Lambda, Lp)
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


def Truncate(atoms: Atoms, f: GraphFuncADMM, coefs: Coefs, F_th, Lambda, Lp):
    terminate = False
    while not terminate:
        remove_idx = remove_atom(atoms, f, coefs, Lambda, Lp)

        new_atoms = deepcopy(atoms)
        new_coefs = deepcopy(coefs)
        new_atoms.remove_atom(new_atoms[remove_idx])
        new_coefs.remove_coef(remove_idx)

        if len(new_coefs) > 0:
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
    f = GraphFuncADMM(graph=graph)
    Lambda = np.random.randn(n, n)
    Lp = np.random.randn(n, n)
    Lp = project_to_bound(Lp)

    iter = 0
    bar = tqdm(total=config.cogent_iters)
    f_values = []
    diff_vals = []
    Lambda_terms = []
    for itr in range(config.cogent_iters):
        sig = atoms.calc_sig(coefs)
        f_sig = f(sig, Lambda, Lp)
        coefs = Forward_step(atoms, f, coefs, Lambda, Lp)

        coefs = Enhancemnet_step(atoms, f, coefs, Lambda, Lp)

        sig = atoms.calc_sig(coefs)
        f_sig_new = f(sig, Lambda, Lp)
        F_th = config.eta * f_sig + (1 - config.eta) * f_sig_new

        atoms, coefs = Truncate(atoms, f, coefs, F_th, Lambda, Lp)

        # min Lp
        CoGEnT_sig = atoms.calc_sig(coefs)
        Lp += config.alpha * (Lambda + config.beta * (CoGEnT_sig.L - Lp))
        Lp = project_to_bound(Lp)

        # max Lambda
        Lambda += config.beta * (CoGEnT_sig.L - Lp)

        d = CoGEnT_sig.L - Lp
        diff = matrix_dot_product(d, d)

        lambda_term = matrix_dot_product(Lambda, d)

        f_sig2 = f(CoGEnT_sig)

        bar.set_postfix(
            {
                "f": f_sig2,
                "diff": diff,
                "lambda_term": lambda_term,
            }
        )
        bar.update()
        f_values.append(f_sig2)
        diff_vals.append(diff)
        Lambda_terms.append(lambda_term)

a = [
    str(np.round(atom.e, 2)) + str(np.round(atom.b, 2)) + str(c)
    for c, atom in zip(coefs, atoms)
]
with open("./file.txt", mode="w") as fid:
    fid.write("\n".join(str(year) for year in a))

plt.plot(f_values[1:])
plt.plot(diff_vals[1:])
plt.plot(Lambda_terms[1:])
plt.show()

# print(coefs.coefs)
