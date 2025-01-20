import json
import pickle as pkl

import numpy as np
from tqdm import tqdm

from Atom.atoms import Atoms
from Atom.coefs import Coefs
from Graph_Atom.graph_atom import GraphAtom
from Graph_Atom.graph_func import GraphFunc
from utils.utils import *
from utils.algorithm_utils_orig import *

path = "results/20250119_221156/"

atoms_path = f"{path}atoms.pkl"
coefs_path = f"{path}coefs.pkl"
graph_path = f"{path}graph.pkl"
Lp_path = f"{path}Lp.pkl"
Lambda_path = f"{path}Lambda.pkl"

with open(atoms_path, "rb") as fid:
    atoms = pkl.load(fid)
with open(coefs_path, "rb") as fid:
    coefs = pkl.load(fid)
with open(graph_path, "rb") as fid:
    graph = pkl.load(fid)
with open(Lp_path, "rb") as fid:
    Lp = pkl.load(fid)
with open(Lambda_path, "rb") as fid:
    Lambda = pkl.load(fid)

config.tau = graph.num_nodes


f = GraphFunc(graph=graph)
f.set_ADMM_params(Lambda, Lp)

sig = atoms.calc_sig(coefs)
f_sig = f(sig, ADMM=False)
print(f_sig)

bar = tqdm(total=config.cogent_iters)
f_values = []
accuracies = []
diff_vals = []
Lambda_terms = []
num_atoms = []
for ii in range(config.cogent_iters):
    for itr in range(config.local_iter):
        atoms, coefs = Forward_step(atoms, f, coefs, sig, ii * config.local_iter + itr)
        # atoms, coefs = check_constraint(atoms, coefs)
        # if sum(coefs) >= n:
        #     atoms, coefs = Enhance_step(atoms, coefs)

        coefs = Enhancemnet_step(atoms, f, coefs, ADMM=False)

        sig = atoms.calc_sig(coefs)
        f_sig_new = f(sig, ADMM=False)
        F_th = config.eta * f_sig + (1 - config.eta) * f_sig_new

        atoms, coefs = Truncate(atoms, f, coefs, F_th)
        # if len(num_atoms) > 2:
        #     atoms, coefs = Matrix_Truncate(atoms, f, coefs, F_th)
        sig = atoms.calc_sig(coefs)

    if config.ADMM:
        # min Lp
        # Lp += config.alpha * (Lambda + config.beta * (CoGEnT_sig.L - Lp))
        Lp = Lambda / config.beta + sig.L
        Lp = project_to_bound(Lp)

        # max Lambda
        d = sig.L - Lp
        Lambda += config.beta * d

        diff = matrix_dot_product(d, d)

        lambda_term = matrix_dot_product(Lambda, d)

        f.set_ADMM_params(Lambda, Lp)
    f_sig = f(sig, ADMM=False)

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
    }

    if config.ADMM:
        results["diff"] = diff
        results["lambda term"] = lambda_term

    bar.set_postfix(results)
    bar.update()
    f_values.append(f_sig)
    num_atoms.append(len(atoms))
    accuracies.append(acc)

    if config.ADMM:
        diff_vals.append(diff)
        Lambda_terms.append(lambda_term)

CoGEnT_sig = atoms.calc_sig(coefs)
results = {
    f"atom {id}": {
        "e": str(np.round(atom.e, 2)),
        "b": str(np.round(atom.b, 2)),
        "R": str(np.round(np.linalg.eigvalsh(atom.R), 2)),
        "λ": str(np.round(c, 2)),
    }
    for id, (c, atom) in enumerate(zip(coefs, atoms))
}

mask = np.argmax(CoGEnT_sig.pi, axis=1) == graph.y
# Replace True with ✔ and False with ✘
marks = np.array(["✔" if value else "✘" for value in mask])[:, None]

LOGGER.info(json.dumps(results, indent=4))
LOGGER.info(np.hstack((np.round(CoGEnT_sig.pi, 2), marks)))

plot_adj(CoGEnT_sig.L)

y_pred = np.argmax(CoGEnT_sig.pi, axis=1)
y_pred_test = y_pred[graph.test_mask]
if y_pred_test.shape[0] > 0:
    acc = calc_accuracy(y_pred_test, graph.y[graph.test_mask])
else:
    acc = 0

plt.figure()
plt.plot(f_values)
# plt.yscale("log")
plt.savefig(f"{save_path}loss.png")
plt.figure()
plt.plot(num_atoms)
plt.savefig(f"{save_path}num_atoms.png")
plt.figure()
plt.plot(accuracies)
plt.title("Accuracy")
plt.savefig(f"{save_path}accuracies.png")

plt.figure()
for id, atom in enumerate(atoms):
    plt.plot(atom.e, "*-", label=f"atom {id}")
plt.legend()
plt.title("Eigenvectors")
plt.savefig(f"{save_path}eigenvectors.png")

filename = f"{save_path}atoms.pkl"
with open(filename, "wb") as fid:
    pkl.dump(atoms, fid)

filename = f"{save_path}coefs.pkl"
with open(filename, "wb") as fid:
    pkl.dump(coefs, fid)
filename = f"{save_path}graph.pkl"
with open(filename, "wb") as fid:
    pkl.dump(graph, fid)

if config.ADMM:
    filename = f"{save_path}Lambda.pkl"
    with open(filename, "wb") as fid:
        pkl.dump(Lambda, fid)

    filename = f"{save_path}Lp.pkl"
    with open(filename, "wb") as fid:
        pkl.dump(Lp, fid)

LOGGER.info(acc)


# ## Initialization
# atoms = Atoms()
# coefs = Coefs()

# E = graph.E
# # E[np.abs(E) < 0.00001] = 0.00001
# R = graph.Ri_matrices
# b = graph.b
# c = graph.num_clusters

# for i in range(c):
#     atom0 = GraphAtom()
#     S, U = np.linalg.eigh(R[i])
#     mask = S >= config.sig_rho_max - 0.0000001
#     S[mask] = config.rho_max - config.rho_min
#     S[~mask] = 0

#     RR = U @ np.diag(S) @ U.T
#     atom0.set_params(E[:, i], RR, b[i])
#     atoms.add_atom(atom0)
#     coefs.add_coef(graph.λ[i])

# sig = atoms.calc_sig(coefs)
# f_sig = f(sig)
# print(f_sig)
