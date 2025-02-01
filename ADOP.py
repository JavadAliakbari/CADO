import numpy as np
from sklearn.metrics.cluster import adjusted_rand_score
from tqdm import tqdm

from Graph_Atom.graph_func_ADOP import GraphFunc_ADOP
from utils.algorithm_utils_orig import *
from utils.SBM import SBM
from utils.utils import *


def claulate_grad_coefs_ADOP(atoms, coefs, graph, grad_sig):
    grad_coefs = []
    for atom, coef in zip(atoms, coefs):

        tmp0 = graph.X @ np.linalg.inv(atom.R) @ np.transpose(graph.X)
        tmp1 = (-1 / (coef**2)) * np.sum(np.dot(atom.e**2, tmp0))
        tmp2 = np.sum(atom.e**2) * np.trace(atom.R)
        grad1 = 2 * config.mu1 * (tmp1 + tmp2)

        tmp3 = np.einsum("i, ij-> j", atom.e**2, grad_sig.pi)
        grad2 = -2 * config.mu2 * np.sum(np.dot(tmp3, atom.b))

        grad3 = config.mu0

        grad = grad1 + grad2 + grad3
        grad_coefs.append(grad)
    return np.array(grad_coefs)


def update_coefs_ADOP(coefs, grad_coefs, lr):
    new_coefs = Coefs()
    for i, coef in enumerate(coefs):
        coef -= lr * grad_coefs[i]
        new_coefs.add_coef(coef)
    return new_coefs


def ADOP(graph, atoms=None, coefs=None, admm=None, log_results=False):
    n = graph.num_nodes
    num_features = graph.num_features
    c = graph.num_clusters

    ## Initialization
    if atoms is None:
        atoms, coefs = create_random_atoms(n, num_features, c, c)
    atoms_opt, coefs_opt = create_optimal_atoms(graph)
    num_atoms = len(atoms)

    #### start from the optimal atoms and coefficients
    # coefs = coefs_opt
    # atoms = atoms_opt

    sig = atoms.calc_sig(coefs)
    opt_sig = atoms_opt.calc_sig(coefs_opt)
    # sig = EmptySig(n, num_features, c)
    f = GraphFunc_ADOP(graph=graph, coefs=coefs)
    f_opt = f(opt_sig, ADMM=False)
    f_sig = f(sig, ADMM=False)
    L_current = deepcopy(sig.L)

    # if log_results:
    bar = tqdm(total=config.cogent_iters, leave=False)
    f_values = [f_sig]
    accuracies = []
    for ii in range(config.cogent_iters):

        #### do alternating minimization
        for itr in range(config.local_iter):
            #### update all atoms together not one by one
            grad = f.calc_grad_ADOP(sig)
            old_atoms = deepcopy(atoms)
            for atom, coef in zip(atoms, coefs):
                atom.update_atom_ADOP(grad, config.lr, coef, graph)

            #### based on what we had before
            # grad_without_L = deepcopy(grad)
            # grad_without_L.L = np.zeros_like(grad.L)
            # grad_coefs = np.array(
            #     [grad_without_L.dot(atom.calc_sig_term()) for atom in old_atoms]
            # )

            grad_coefs = claulate_grad_coefs_ADOP(old_atoms, coefs, graph, grad)

            coefs = update_coefs_ADOP(coefs, grad_coefs, config.lr)
            sig = atoms.calc_sig(coefs)
            sig.L = L_current - config.lr * grad.L

        ##### do alternating projection
        for itr in range(config.local_iter_ALP):
            #### project to L=Aλ
            A = np.array(
                [np.outer(atom.e, atom.e) for atom in atoms]
            )  # shape = (num_atoms, n, n)
            B = np.einsum("ijk,ljk->il", A, A)  # shape = (num_atoms, num_atoms)
            y_tmp = np.einsum("kij,k->ij", A, coefs) - sig.L  # shape = (n, n)
            y_bar = np.array([np.einsum("ij,ij->", A[i], y_tmp) for i in range(c)])
            tmp = np.linalg.inv(B + np.eye(num_atoms))
            alpha = np.matmul(tmp, y_bar)  # shape = (num_atoms)
            lagrange_z = y_tmp - np.einsum("kij,k->ij", A, alpha)

            sig.L += lagrange_z
            coefs -= np.einsum("ijk,jk->i", A, lagrange_z)

            #### project coefficients to be non-negative
            coefs = np.maximum(coefs, 0)
            #### project L to the box
            L_current = deepcopy(sig.L)
            sig = atoms.calc_sig(Coefs(coefs))
            sig.L = project_to_bound(L_current)

        f_sig = f(sig, ADMM=False)
        f_values.append(f_sig)

        # f_sig = f(sig, ADMM=False)

        y_pred = np.argmax(sig.pi, axis=1)
        y_pred_test = y_pred[graph.test_mask]
        if y_pred_test.shape[0] > 0:
            acc = calc_accuracy(y_pred_test, graph.y[graph.test_mask])
        else:
            acc = 1

        results = {
            f"f": f"{f_sig:,.4f}",
            "f_opt": f"{f_opt:,.4f}",
            "acc": acc,
        }

        # if log_results:
        bar.set_postfix(results)
        bar.update()
        f_values.append(f_sig)
        accuracies.append(acc)

    return (
        atoms,
        coefs,
        graph,
        results,
        f_values,
        accuracies,
        f_opt,
        # admm,
    )


if __name__ == "__main__":
    samples_per_cluster = [100, 100, 100]  # Number of training samples in each cluster
    num_features = 6  # Dimension of the features at each node
    num_signal_features = 2
    n = sum(samples_per_cluster)
    c = len(samples_per_cluster)
    n_clusters = c

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

    cluster_labels = run_spectral_clustering(graph, n_clusters)
    ari = adjusted_rand_score(graph.y, cluster_labels)
    # print(f"ARI: {ari}")

    (atoms, coefs, graph, results, f_values, accuracies, f_opt) = ADOP(graph)
    # ) = CANO(graph, atoms, coefs, admm)

    # log_results(atoms, coefs, graph, results, save_path)
    plot_results2(f_values, accuracies, atoms, save_path, f_opt)
    save_model_params2(atoms, coefs, graph, save_path)
