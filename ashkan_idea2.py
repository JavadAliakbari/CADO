import numpy as np
from tqdm import tqdm

from Graph_Atom.ADMM import ADMM_cls
from Graph_Atom.graph_func import GraphFunc
from utils.algorithm_utils_orig import *
from utils.SBM import SBM
from utils.utils import *

# Set print options to show the entire array
np.set_printoptions(threshold=np.inf, suppress=False)

if __name__ == "__main__":
    #### CoGEnT algorithm
    samples_per_cluster = (
        config.samples_per_cluster
    )  # Number of training samples in each cluster
    num_features = config.dimension  # Dimension of the features at each node
    num_signal_features = config.signal_subspace_dim
    n = config.num_nodes
    c = config.num_clusters
    config.tau = n
    graph = SBM(
        samples_per_cluster,
        config.p,
        config.q,
        num_features,
        num_signal_features,
        # train_ratio=1,
        train_ratio=config.train_ratio,
    )
    plot_adj(graph.A, name="A")

    E = graph.E
    R = graph.Ri_matrices
    b = graph.b
    opt_sig = create_sig(E, R, b, graph.λ)

    ## Initialization
    atoms, coefs = create_random_atoms(n, num_features, c, c)
    atoms_opt, coefs_opt = create_optimal_atoms(graph)
    num_atoms = len(atoms)

    sig = atoms.calc_sig(coefs)
    # sig = EmptySig(n, num_features, c)
    f = GraphFunc(graph=graph)
    if config.ADMM == "L":
        admm = ADMM_cls(sig.L, project_to_bound)
        f.set_ADMM_params(admm)
    elif config.ADMM == "pi":
        admm = ADMM_cls(sig.pi, project_to_simplex)
        f.set_ADMM_params(admm)
    else:  # config.ADMM == "None"
        admm = None

    f_opt = f(opt_sig)
    f_sig = f(sig)

    bar = tqdm(total=config.cogent_iters)
    f_values = []
    accuracies = []
    diff_vals = []
    Lambda_terms = []
    num_atoms_list = []
    diff = 0
    Lambda_term = 0
    for ii in range(config.cogent_iters):
        for itr in range(config.local_iter):
            # atom_idx = np.random.choice(num_atoms, 1)[0]
            # atom = atoms[atom_idx]
            # coef = coefs[atom_idx]
            idx = np.arange(num_atoms)
            np.random.shuffle(idx)
            atoms.rearnge(idx)
            coefs.rearnge(idx)
            for atom, coef in zip(atoms, coefs):
                sig2 = sig - coef * atom.calc_sig_term()
                grad = f.calc_grad(sig2)
                # grad_atom = atom.calc_grad(grad)
                # atom.update(grad_atom, 2 * coef * 0.1)  # Update atom
                atom.conditional_gradient_update(grad, ii * config.local_iter + itr)
                sig = atoms.calc_sig(coefs)  # Update signal
                f_sig = f(sig)
                f_values.append(f_sig)
                # atoms, coefs = update_coefs(atoms, f, coefs, sig, 0.2, 20)  # Update coefs
            coefs = Enhancemnet_step(atoms, f, coefs)

        if admm is not None:
            if config.ADMM == "L":
                diff, lambda_term, lambda_norm = admm.ADMM_update(sig.L)
            elif config.ADMM == "pi":
                diff, lambda_term = admm.ADMM_update(sig.pi)
            diff_vals.append(diff)
            Lambda_terms.append(lambda_term)
        f_sig = f(sig)

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
        }

        bar.set_postfix(results)
        bar.update()
        f_values.append(f_sig)
        num_atoms_list.append(len(atoms))
        accuracies.append(acc)

        # if ii % 100000 == 0:
        #     log_results(atoms, coefs, graph, results, save_path, terminal=False)
        #     plot_results(f_values, num_atoms, accuracies, atoms, save_path)

    # log_results(atoms, coefs, graph, results, save_path)
    plot_results22(f_values, accuracies, save_path, f_opt=None)
    # save_model_params(atoms, coefs, graph, admm, save_path)
