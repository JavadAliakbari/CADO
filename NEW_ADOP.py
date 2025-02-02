import warnings

import numpy as np
from sklearn.metrics.cluster import adjusted_rand_score
from tqdm import tqdm

from utils.SBM import SBM
from utils.utils import *

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="Graph is not fully connected, spectral embedding may not work as expected.",
)

###############################################################
###############################################################


def ADOP(graph):

    #### Initialization
    (
        atoms_e,
        atoms_R,
        atoms_b,
        coefs,
        atoms_e_opt,
        atoms_R_opt,
        atoms_b_opt,
        coefs_opt,
        sig_L,
    ) = initialize_atoms(graph, init=config.initialize_atoms)

    f_sig = compute_objective_value(graph, atoms_e, atoms_R, atoms_b, coefs)
    f_opt = compute_objective_value(
        graph, atoms_e_opt, atoms_R_opt, atoms_b_opt, coefs_opt
    )

    f_values = [f_sig]
    results = []
    accuracies = []

    bar = tqdm(total=config.total_iterations, leave=True)
    for outer_iter in range(config.total_iterations):

        for itr in range(config.local_iter_ALM):
            atoms_e, atoms_R, atoms_b, coefs, sig_L = update_atoms_and_L_ADOP(
                graph, atoms_e, atoms_R, atoms_b, coefs, sig_L
            )
            # print(f"iter {outer_iter} - coefs: {coefs}")

        for itr in range(config.local_iter_ALP):
            sig_L, coefs = project_sig_L_and_coefs_ADOP(atoms_e, sig_L, coefs)
            # print(f"iter {outer_iter} - coefs: {coefs}")

        f_values.append(
            compute_objective_value(
                graph, atoms_e, atoms_R, atoms_b, coefs, defined_sig_L=sig_L
            )
        )

        acc = calc_accuracy_ADOP(graph, atoms_e, atoms_R, atoms_b, coefs)
        accuracies.append(acc)

        results = {
            f"f": f"{f_values[-1]:,.4f}",
            "f_opt": f"{f_opt:,.4f}",
            "acc": acc,
        }

        bar.set_postfix(results)
        bar.update()

    return (
        atoms_e,
        atoms_R,
        atoms_b,
        coefs,
        atoms_e_opt,
        atoms_R_opt,
        atoms_b_opt,
        coefs_opt,
        f_opt,
        f_values,
        accuracies,
        results,
        sig_L,
    )


if __name__ == "__main__":

    graph = SBM(
        config.samples_per_cluster,
        config.p,
        config.q,
        config.dimension,
        config.signal_subspace_dim,
        train_ratio=config.train_ratio,
    )

    (
        atoms_e,
        atoms_R,
        atoms_b,
        coefs,
        atoms_e_opt,
        atoms_R_opt,
        atoms_b_opt,
        coefs_opt,
        f_opt,
        f_values,
        accuracies,
        results,
        sig_L,
    ) = ADOP(graph)

    log_results(sig_L,atoms_e, atoms_R, atoms_b, coefs, graph, results, save_path)
    plot_results2(f_values, accuracies, atoms_e, save_path, f_opt,)
    # save_model_params2(atoms, coefs, graph, save_path)
