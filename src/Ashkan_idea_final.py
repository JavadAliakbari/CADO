import numpy as np
from tqdm import tqdm

from Graph_Atom.ADMM import ADMM_cls
from soft_thresholding import soft_thresholdinhg
from utils.SBM import SBM
from Graph_Atom.graph_func import GraphFunc

from utils.algorithm_utils_orig import *
from utils.utils import *

# Set print options to show the entire array
np.set_printoptions(threshold=np.inf, suppress=False)


def proximal_conditional_descent(
    atoms: Atoms, coefs: Coefs, f: GraphFunc, num_atoms, iter=0
):
    """
    Proximal gradient descent algorithm
    """
    # atoms, coefs = create_random_atoms(sig.n, sig.d, sig.c, num_atoms)
    for itr in range(config.local_iter):
        idx = np.arange(num_atoms)
        np.random.shuffle(idx)
        sig = atoms.calc_sig(coefs)  # Update signal
        for ind in idx:
            sig2 = sig - coefs[ind] * atoms[ind].calc_sig_term()
            grad = f.calc_grad(sig2)
            atoms[ind].conditional_gradient_update(grad, iter * config.local_iter + itr)
        if iter % 1 == 0:
            coefs = Enhancemnet_step(atoms, f, coefs)

    return atoms, coefs


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

    ## Initialization
    atoms, coefs = create_random_atoms(n, num_features, c, c)
    atoms_opt, coefs_opt = create_optimal_atoms(graph)
    num_atoms = len(atoms)

    atoms = atoms_opt
    coefs = coefs_opt

    sig = atoms.calc_sig(coefs)
    # sig = EmptySig(n, num_features, c)
    f = GraphFunc(graph=graph)
    admm = None

    f_opt = f(opt_sig, ADMM=False)
    f_sig = f(sig)

    bar = tqdm(total=config.iters)
    f_values = []
    accuracies = []
    diff_vals = []
    Lambda_terms = []
    Lambda_norms = []
    components = []
    num_atoms_list = []
    epsilsons = []
    diff = 0
    Lambda_term = 0

    if config.Abar:
        A_orig = deepcopy(graph.Abar)
    else:
        A_orig = deepcopy(graph.A)

    A = np.zeros((n, n))

    for ii in range(config.iters):

        # sig = atoms.calc_sig(coefs)
        totaliter = 10
        # if ii > 0:
        #     A = 0.5 * A + 0.5 * sig.L
        # else:
        #     A = sig.L
        for _ in range(totaliter):
            A = soft_thresholdinhg(A_orig, A, lr=0.01, mu=1)
        graph.A = A

        atoms, coefs = proximal_conditional_descent(atoms, coefs, f, num_atoms, iter=ii)

        f_sig, L_term, R_term, pi_term, ADMM_term = f(sig, ADMM=False, components=True)

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
            "diff": L_term,
            "f_opt": f"{f_opt:,.4f}",
        }

        bar.set_postfix(results)
        bar.update()
        f_values.append(f_sig)
        num_atoms_list.append(len(atoms))
        accuracies.append(acc)
        epsilsons.append([atoms.e.copy() for atoms in atoms])
        components.append([L_term, R_term, pi_term, ADMM_term])

        # if ii % 100000 == 0:
        #     log_results(atoms, coefs, graph, results, save_path, terminal=False)
        #     plot_results(f_values, num_atoms, accuracies, atoms, save_path)

    log_results(atoms, coefs, graph, results, save_path)
    plot_results(
        f_values,
        accuracies,
        Lambda_norms,
        atoms,
        epsilsons,
        components,
        admm,
        save_path,
        f_opt,
    )
    save_model_params(atoms, coefs, graph, admm, save_path)
