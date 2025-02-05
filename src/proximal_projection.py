import numpy as np
from tqdm import tqdm
from utils.algorithm_utils_orig import *
from utils.SBM import SBM
from utils.utils import *

# Set print options to show the entire array
np.set_printoptions(threshold=np.inf, suppress=False)


def dot(sig1: GraphSig, sig2: GraphSig):
    term1 = np.sum(sig1.L * sig2.L, axis=1)
    if sig1.Rv is not None and sig2.Rv is not None:
        term2 = np.einsum("ijk, ijk->i", sig1.Rv, sig2.Rv)
    else:
        term2 = 0
    term3 = np.sum(sig1.pi * sig2.pi, axis=1)

    return term1 + term2 + term3


def GD(atoms, coefs: Coefs, f, lr=0.01, mu0=1):
    """
    Gradient Descent algorithm
    """
    sig = atoms.calc_sig(coefs)
    grad = f.calc_grad(sig)
    sig.update(grad, lr)

    Z = lr * grad
    g = np.array([-dot(Z, atom.calc_sig_term()) + mu0 * lr / 2 for atom in atoms]).T
    Piv = np.array([coef * atom.e**2 for atom, coef in zip(atoms, coefs)]).T
    Piv -= lr * g
    Piv = np.maximum(Piv, 0)
    c = np.sum(Piv, axis=0)

    coefs.update_coefficients(c)
    # E2 = Piv @ np.diag(1 / c)
    # E = np.sqrt(E2)

    # for i in range(len(atoms)):
    #     atoms[i].e = E[:, i]

    for atom in atoms:
        atom.CG_update_step(Z)

    return atoms, coefs


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

    f = GraphFunc(graph)

    atoms_opt, coefs_opt = create_optimal_atoms(graph)
    opt_sig = atoms_opt.calc_sig(coefs_opt)
    L_opt = opt_sig.L

    L = deepcopy(opt_sig.L)
    coefs = Coefs(np.ones(n))
    atoms = Atoms()
    for i in range(n):
        atom = GraphAtom()
        e = np.zeros(n)
        e[i] = 1

        R = np.outer(graph.X[i], graph.X[i])
        b = np.zeros(c)
        b[graph.y[i]] = 1

        atom.set_params(e, R, b)

        atoms.add_atom(atom)

    # L = np.zeros((n, n))
    diff_list = []  # List to store the difference between L and L_opt
    total_iter = 10
    bar = tqdm(total=total_iter, leave=True)
    for i in range(total_iter):
        # print(f"Iteration {i}")
        L = atoms.calc_sig(coefs).L
        # plot_adj(L, name=f"L{i}")
        atoms, coefs = GD(atoms, coefs, f, lr=0.01, mu0=2)
        sum_c = np.sum(coefs.get_coefs())
        L = atoms.calc_sig(coefs).L
        diff = np.linalg.norm(L - L_opt)
        bar.set_postfix_str({"Diff": diff, "Sum c": sum_c})
        bar.update(1)
        diff_list.append(diff)

    plot_adj(L, name="L")
    LOGGER = getLOGGER(
        name=f"{now}",
        log_on_file=True,
        save_path=save_path,
    )
    sig = atoms.calc_sig(coefs)
    results = {
        f"atom {id}": {
            "e": str(np.round(atom.e, 2)),
            "b": str(np.round(atom.b, 2)),
            "R": str(np.round(np.linalg.eigvalsh(atom.R), 2)),
            "λ": str(np.round(c, 2)),
        }
        for id, (c, atom) in enumerate(zip(coefs, atoms))
    }

    mask = np.argmax(sig.pi, axis=1) == graph.y
    # Replace True with ✔ and False with ✘
    marks = np.array(["✔" if value else "✘" for value in mask])[:, None]

    LOGGER.info(json.dumps(results, indent=4))
    LOGGER.info(np.hstack((np.round(sig.pi, 2), marks)))

    plt.figure()
    plt.plot(diff_list)
    plt.xlabel("Iteration")
    plt.ylabel("Difference")
    plt.title("Difference between L and L_opt")
    plt.savefig(f"{save_path}soft_thresholding_diff.png")
    # plt.show()
