import numpy as np
from tqdm import tqdm
from utils.algorithm_utils_orig import *
from utils.SBM import SBM
from utils.utils import *

# Set print options to show the entire array
np.set_printoptions(threshold=np.inf, suppress=False)


def dot(sig1: GraphSig, atom: GraphAtom):
    term2 = np.einsum("ijk, jk->i", sig1.Rv, atom.R)
    term3 = np.einsum("ij,j->i", sig1.pi, atom.b)

    return term2 + term3


def update_R(atom, grad: GraphSig, Pv):
    # grad_R = atom.calc_grad_Ri(grad)
    grad_R = np.einsum("ijk,i->jk", grad.Rv, Pv)

    S, U = np.linalg.eigh(grad_R)

    mask1 = S > 0
    S[mask1] = config.rho_max - config.rho_min
    S[~mask1] = 0
    atom.R = U @ np.diag(S) @ U.T


## Step 3: Fix epsilon and R, update b (closed-form solution)
def update_b(atom, grad: GraphSig, Pv):
    # a = atom.calc_grad_bi(grad)
    a = np.einsum("i, ij-> j", Pv, grad.pi)
    atom.b = np.zeros(grad.pi.shape[-1], dtype=float)
    atom.b[np.argmax(a)] = 1


def calc_sig(atoms, Piv):
    sig = EmptySig(atoms[0].n, atoms[0].d, atoms[0].c)
    for atom, Pi in zip(atoms, Piv.T):
        sig.Rv += np.einsum("i,jk->ijk", Pi, atom.R)
        sig.pi += np.einsum("i,j->ij", Pi, atom.b)

    return sig

    # sig.Rv += config.rho_min * np.eye(sig.Rv.shape[-1])[None, :]


def proximal_projection(sig, lr=0.00001, mu0=1):
    """
    Proximal projection algorithm
    """
    atoms, coefs = create_random_atoms(n, num_features, sig.c, 20)
    Piv = np.array([coef * atom.e**2 for atom, coef in zip(atoms, coefs)]).T

    t2 = 0.0001
    total = 100
    bar = tqdm(total=total, leave=False)
    Z = EmptySig(n, sig.d, sig.c)
    for i in range(100):
        # if i == 0:
        #     Z = sig - atoms.calc_sig(coefs)
        # else:
        Z = (1 - t2) * Z + t2 * (sig - calc_sig(atoms, Piv))

        g = np.array([-dot(Z, atom) + mu0 * lr / 2 for atom in atoms]).T
        Piv -= lr * g
        Piv = np.maximum(Piv, 0)

        # for i in range(len(atoms)):
        #     atoms[i].e = E[:, i]

        # Remove atoms and coefficients corresponding to zero coefficients
        # new_atoms = Atoms()
        # new_coefs = Coefs()

        # non_zero_indices = [i for i, coef in enumerate(coefs) if coef > 1e-5]
        # [new_atoms.add_atom(atoms[i]) for i in non_zero_indices]
        # new_coefs.update_coefficients([coefs[i] for i in non_zero_indices])
        # atoms = new_atoms
        # coefs = new_coefs

        for ind in range(len(atoms)):
            # atom.CG_update_step(Z)
            # atom.update_R(Z)
            # atom.update_b(Z)
            update_R(atom, Z, Piv[:, ind])
            update_b(atom, Z, Piv[:, ind])

        diff = sig - calc_sig(atoms, Piv)
        bar.update(1)
        bar.set_postfix_str(
            {"norm Z": np.linalg.norm(diff.Rv), "atomic norm": np.sum(Piv)}
        )

    atomic_sig = calc_sig(atoms, Piv)
    c = np.sum(Piv, axis=0)
    E2 = Piv @ np.diag(1 / (c + 1e-10))
    E = np.sqrt(E2)
    atomic_sig.L = E @ np.diag(c) @ E.T

    return atomic_sig
    # return atoms, coefs


def GD(sig, f, lr=0.01, mu0=1):
    """
    Gradient Descent algorithm
    """
    # sig = atoms.calc_sig(coefs)
    grad = f.calc_grad(sig)
    sig.update(grad, lr)

    new_sig = proximal_projection(sig, lr, mu0)

    return new_sig


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
    f_opt, l_opt, R_opt, pi_opt, ADMM_opt = f(opt_sig, components=True)
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
    total_iter = 100
    sig = atoms.calc_sig(coefs)
    bar = tqdm(total=total_iter, leave=True)
    for i in range(total_iter):
        # print(f"Iteration {i}")
        # plot_adj(L, name=f"L{i}")
        sig = GD(sig, f, lr=0.01, mu0=1)
        # sum_c = np.sum(coefs.get_coefs())
        # sig = atoms.calc_sig(coefs)
        L = sig.L
        diff = np.linalg.norm(L - L_opt)
        f_sig, L_term, R_term, pi_term, ADMM = f(sig, components=True)
        bar.set_postfix_str({"Diff": diff, "R": R_term, "R_opt": R_opt})
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
