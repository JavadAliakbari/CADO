import numpy as np
from tqdm import tqdm

from utils.SBM import SBM
from Graph_Atom.graph_func import GraphFunc
from Graph_Atom.ADMM import ADMM_cls
from utils.algorithm_utils_orig import *
from utils.utils import *

# Set print options to show the entire array
np.set_printoptions(threshold=np.inf, suppress=False)

if __name__ == "__main__":
    #### CoGEnT algorithm
    samples_per_cluster = [100, 100, 100]  # Number of training samples in each cluster
    num_features = 6  # Dimension of the features at each node
    num_signal_features = 2
    n = sum(samples_per_cluster)
    c = len(samples_per_cluster)
    probs = np.full((c, c), config.q)
    np.fill_diagonal(probs, config.p)
    # probs = np.eye(c)
    config.tau = n
    graph = SBM(
        samples_per_cluster,
        probs,
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

    E = graph.E
    # E[np.abs(E) < 0.00001] = 0.00001
    R = graph.Ri_matrices
    b = graph.b

    ## Initialization
    # atoms, coefs = create_random_atoms()
    atoms, coefs = create_optimal_atoms(graph)
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
    Lambda_norms = []
    num_atoms = []
    epsilsons = []
    for ii in range(config.cogent_iters):
        for itr in range(config.local_iter):
            for atom, coef in zip(atoms, coefs):
                grad = f.calc_grad(sig)
                grad_atom = atom.calc_grad(grad)
                atom.update(grad_atom, 2 * coef * config.lr)  # Update atom
                sig = atoms.calc_sig(coefs)  # Update signal
                f_sig = f(sig)
                f_values.append(f_sig)
            # atoms, coefs = update_coefs(atoms, f, coefs, sig, 0.2, 20)  # Update coefs
            coefs = Enhancemnet_step(atoms, f, coefs)

        if admm is not None:
            if config.ADMM == "L":
                diff, lambda_term, Lambda_norm = admm.ADMM_update(sig.L)
            elif config.ADMM == "pi":
                diff, lambda_term, Lambda_norm = admm.ADMM_update(sig.pi)
            diff_vals.append(diff)
            Lambda_terms.append(lambda_term)
            Lambda_norms.append(Lambda_norm)
        else:
            diff = 0
            lambda_term = 0
            Lambda_norm = 0
        f_sig = f(sig)

        y_pred = np.argmax(sig.pi, axis=1)
        y_pred_test = y_pred[graph.test_mask]
        if y_pred_test.shape[0] > 0:
            acc = calc_accuracy(y_pred_test, graph.y[graph.test_mask])
        else:
            acc = 1

        results = {
            "num atoms": len(atoms),
            f"f": f"{f_sig:,.4f}",
            "acc": acc,
            "diff": diff,
            "lambda term": lambda_term,
            "lambda norm": Lambda_norm,
            "f_opt": f"{f_opt:,.4f}",
        }

        if log_results:
            bar.set_postfix(results)
            bar.update()
        f_values.append(f_sig)
        num_atoms.append(len(atoms))
        accuracies.append(acc)
        epsilsons.append([atoms.e.copy() for atoms in atoms])

    log_results(atoms, coefs, graph, results, save_path)
    plot_results(
        f_values,
        accuracies,
        Lambda_norms,
        atoms,
        epsilsons,
        admm,
        save_path,
        f_opt,
    )
    save_model_params(atoms, coefs, graph, admm, save_path)
