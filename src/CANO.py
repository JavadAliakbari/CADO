import numpy as np
from tqdm import tqdm

from Graph_Atom.ADMM import ADMM_cls
from Graph_Atom.graph_func import GraphFunc
from utils.algorithm_utils_orig import *
from utils.utils import *


def CANO(graph, atoms=None, coefs=None, admm=None, log_results=True):
    n = graph.num_nodes
    num_features = graph.num_features
    c = graph.num_clusters
    ## Initialization
    if atoms is None:
        atoms, coefs = create_random_atoms(n, num_features, c, c)
    atoms_opt, coefs_opt = create_optimal_atoms(graph)
    num_atoms = len(atoms)
    coefs = coefs_opt
    # atoms = atoms_opt

    sig = atoms.calc_sig(coefs)
    opt_sig = atoms_opt.calc_sig(coefs_opt)
    # sig = EmptySig(n, num_features, c)
    f = GraphFunc(graph=graph)
    f_opt = f(opt_sig)
    if admm is None:
        if config.ADMM == "L":
            admm = ADMM_cls(sig.L, project_to_bound)
            f.set_ADMM_params(admm)
        elif config.ADMM == "pi":
            admm = ADMM_cls(sig.pi, project_to_simplex)
            f.set_ADMM_params(admm)
        else:  # config.ADMM == "None"
            admm = None
    else:
        f.set_ADMM_params(admm)

    f_sig = f(sig, ADMM=False)

    # if log_results:
    bar = tqdm(total=config.cogent_iters, leave=False)
    f_values = [f_sig]
    accuracies = []
    diff_vals = []
    Lambda_terms = []
    if admm is not None:
        Lambda_norms = [np.linalg.norm(admm.Lambda)]
    else:
        Lambda_norms = []
    epsilsons = []
    components = []
    diff = 0
    Lambda_term = 0
    for ii in range(config.cogent_iters):
        for itr in range(config.local_iter):
            # atom_idx = np.random.choice(num_atoms, 1)[0]
            # atom = atoms[atom_idx]
            # coef = coefs[atom_idx]
            idx = np.arange(num_atoms)
            np.random.shuffle(idx)
            # atoms.rearnge(idx)
            # coefs.rearnge(idx)
            sig = atoms.calc_sig(coefs)  # Update signal
            for ind in idx:
                if config.CGD:
                    sig2 = sig - coefs[ind] * atoms[ind].calc_sig_term()
                    # new_atoms = deepcopy(atoms)
                    # new_atoms.remove_atom(new_atoms[ind])
                    # new_coefs = deepcopy(coefs)
                    # new_coefs.remove_coef(ind)
                    # new_c = project_to_simplex(new_coefs, config.tau)
                    # new_coefs.update_coefficients(new_c)
                    # sig2 = new_atoms.calc_sig(new_coefs)
                    grad = f.calc_grad(sig2)
                    # grad_atom = atom.calc_grad(grad)
                    # atom.update(grad_atom, 2 * coef * 0.1)  # Update atom
                    atoms[ind].conditional_gradient_update(
                        grad, ii * config.local_iter + itr
                    )
                else:
                    grad = f.calc_grad(sig)
                    grad_atom = atoms[ind].calc_grad(grad)
                    atoms[ind].update(
                        grad_atom, 2 * coefs[ind] * config.lr
                    )  # Update atom
                sig = atoms.calc_sig(coefs)  # Update signal
                f_sig = f(sig)
                f_values.append(f_sig)
                # atoms, coefs = update_coefs(atoms, f, coefs, sig, 0.2, 20)  # Update coefs
                # coefs = Enhancemnet_step(atoms, f, coefs)
            if ii % 1 == 0:
                coefs = Enhancemnet_step(atoms, f, coefs)

        if admm is not None:
            if config.ADMM == "L":
                diff, lambda_term, lambda_norm = admm.ADMM_update(sig.L)
            elif config.ADMM == "pi":
                diff, lambda_term, lambda_norm = admm.ADMM_update(sig.pi)
            diff_vals.append(diff)
            Lambda_terms.append(lambda_term)
            Lambda_norms.append(lambda_norm)
        else:
            diff = 0
            lambda_term = 0
            lambda_norm = 0

        f_sig, L_term, R_term, pi_term, ADMM_term = f(sig, ADMM=False, components=True)

        y_pred = np.argmax(sig.pi, axis=1)
        y_pred_test = y_pred[graph.test_mask]
        if y_pred_test.shape[0] > 0:
            acc = calc_accuracy(y_pred_test, graph.y[graph.test_mask])
        else:
            acc = 1

        results = {
            f"f": f"{f_sig:,.4f}",
            "acc": acc,
            "diff": diff,
            "lambda term": lambda_term,
            "lambda norm": f"{lambda_norm:,.4f}",
            "f_opt": f"{f_opt:,.4f}",
            # "beta": config.beta,
        }

        # if log_results:
        bar.set_postfix(results)
        bar.update()
        f_values.append(f_sig)
        accuracies.append(acc)
        Lambda_norms.append(lambda_norm)
        epsilsons.append([atoms.e.copy() for atoms in atoms])
        components.append([L_term, R_term, pi_term, ADMM_term])

        # check if the algorithm converged:
        # if ii > 0 and np.abs(f_values[-1] - f_values[-2]) < 1e-4:
        #     if admm is None:
        #         break
        #     else:
        #         if np.abs(Lambda_terms[-1]) < 1e-3 and np.abs(diff_vals[-1]) < 1e-5:
        #             break

    return (
        atoms,
        coefs,
        graph,
        results,
        f_values,
        accuracies,
        f_opt,
        admm,
        Lambda_norms,
        epsilsons,
        components,
    )
