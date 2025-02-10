import numpy as np

from utils.utils import plot_adj

import numpy as np
from tqdm import tqdm

from utils.SBM import SBM
from Graph_Atom.graph_func import GraphFunc
from utils.algorithm_utils_orig import *
from utils.utils import *

from Graph_Atom.general_ADMM import ADMM_sig
from Graph_Atom.graph_func import GraphFunc
from utils.algorithm_utils_orig import *
from utils.utils import *

# Set print options to show the entire array
np.set_printoptions(threshold=np.inf, suppress=False)


def CANO2(graph, atoms=None, coefs=None, admm=None, log_results=True):
    n = graph.num_nodes
    num_features = graph.num_features
    c = graph.num_clusters
    ## Initialization
    if atoms is None:
        atoms, coefs = create_random_atoms(n, num_features, c, c)
    atoms_opt, coefs_opt = create_optimal_atoms(graph)
    coefs = coefs_opt
    # atoms = atoms_opt

    sig = EmptySig(n, num_features, c)
    # sig.L = graph.Abar
    opt_sig = atoms_opt.calc_sig(coefs_opt)
    # sig = EmptySig(n, num_features, c)
    f = GraphFunc(graph=graph)
    f_opt = f(opt_sig)
    admm = ADMM_sig(n, num_features, c, graph)
    f.set_ADMM_params(admm)

    f_sig = f(sig, ADMM=False)

    # if log_results:
    bar = tqdm(total=config.iters, leave=False)
    f_values = [f_sig]
    accuracies = []
    Lambda_norms = []
    epsilsons = []
    components = []
    diff = 0
    # coefs.update_coefficients(coefs_opt.get_coefs())
    for ii in range(config.iters):
        f_sig = f(sig, ADMM=False)
        sig, coefs, atoms = admm.ADMM_update(sig, f)
        diff, lambda_term, lambda_norm = admm.ADMM_metric(sig)

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

    plot_adj(sig.L, name=f"LL")

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
