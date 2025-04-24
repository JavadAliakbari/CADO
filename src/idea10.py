from copy import deepcopy

import numpy as np
from tqdm import tqdm
from scipy.special import softmax

from Graph_Atom.graph_func import GraphFunc
from utils.algorithm_utils import *
from utils.utils import *

# trust_region_max: 0.2                 # Maximum allowed change magnitude
# trust_region_min: 0.02                # Minimum allowed change magnitude
# ema_decay: 0.99                       # Exponential moving average decay rate
# polyak_start: 0.5                     # When to start Polyak averaging (fraction of total iters)
# polyak_weight: 0.9                    # Maximum weight for Polyak averaging


def schedule_lr(
    ii,
    max_iter=algorithm.iters,
    min_lr=optimizer.min,
    max_lr=optimizer.lr,
    warmup_steps=algorithm.warmup_steps,
    decay_steps=optimizer.decay_steps,
    decay_rate=optimizer.decay_rate,
    method=optimizer.method,
    # method="exponential",
    # method="cosine",
):
    if method == "cosine":
        lr = min_lr + 0.5 * (max_lr - min_lr) * (
            1 + np.cos(np.pi * (ii - warmup_steps) / (max_iter - warmup_steps))
        )
        if ii < warmup_steps:
            lr = max_lr + (min_lr - max_lr) * (ii / warmup_steps)
    elif method == "linear":
        lr = max_lr + (min_lr - max_lr) * (ii / max_iter)
        if ii < warmup_steps:
            lr = max_lr + (min_lr - max_lr) * (ii / warmup_steps)
    elif method == "exponential":
        lr = max_lr + (min_lr - max_lr) * (ii / max_iter)
        if ii < warmup_steps:
            lr = max_lr + (min_lr - max_lr) * (ii / warmup_steps)
        elif ii < max_iter - decay_steps:
            lr = max_lr
        else:
            lr = max_lr + (min_lr - max_lr) * np.exp(
                -decay_rate * (ii - max_iter + decay_steps)
            )
    return lr


def adaptive_temperature(
    ii, grad_norm, prev_temp=None, loss_change=None, max_iter=algorithm.iters
):
    """
    Adaptively calculate temperature for soft assignment based on gradient norm and loss change
    """
    s = algorithm.soft_assign_s
    base_temp = max(1e-5, (1.0 - s * ii / max_iter)) * grad_norm

    # If we have previous info, make the temperature adaptive to loss landscape
    if prev_temp is not None and loss_change is not None:
        # If loss is increasing or changing rapidly, increase temperature for more exploration
        if loss_change > 0:
            damping = min(1.5, 1.0 + abs(loss_change))
            return min(prev_temp * damping, base_temp * 2.0)
        else:
            # If loss is decreasing nicely, gradually reduce temperature
            damping = max(0.8, 1.0 - abs(loss_change) * 0.1)
            return max(prev_temp * damping, base_temp * 0.5)

    return base_temp


def update(
    atoms: Atoms,
    coefs: Coefs,
    f,
    ii,
    prev_grad=None,
    update_mu0=False,
    mu0=None,
    mu1=None,
    update_mu1=False,
    prev_mu0_grad=None,
    prev_mu1_grad=None,
    momentum_beta=0.9,
    soft_assignment=True,
    prev_atoms=None,
    prev_coefs=None,
    prev_temp=None,
    prev_loss=None,
    curr_loss=None,
):
    sig = atoms.calc_sig(coefs)
    grad = f.calc_grad(sig)

    if update_mu0:
        grad_mu0 = f.calc_grad_mu0(sig)

        if prev_mu0_grad is not None:
            grad_mu0 = momentum_beta * prev_mu0_grad + (1 - momentum_beta) * grad_mu0
        # Update mu0 using gradient descent
        mu0_lr = schedule_lr(
            ii, max_lr=1e-6, min_lr=1e-12, max_iter=algorithm.iters, method="cosine"
        )  # Can use different learning rate schedule
        # mu0_lr = 1e-6
        new_mu0 = max(1e-4, mu0 - mu0_lr * grad_mu0)  # Ensure mu0 stays positive
        f.update_mu0(new_mu0)
    else:
        new_mu0 = mu0
        grad_mu0 = prev_mu0_grad

    if update_mu1:
        grad_mu1 = f.calc_grad_mu1(sig)
        if prev_mu1_grad is not None:
            grad_mu1 = momentum_beta * prev_mu1_grad + (1 - momentum_beta) * grad_mu1
        mu1_lr = schedule_lr(
            ii, max_lr=1e-6, min_lr=1e-12, max_iter=algorithm.iters, method="cosine"
        )  # Can use different learning rate schedule
        new_mu1 = max(1e-4, mu1 - mu1_lr * grad_mu1)  # Ensure mu1 stays positive
        f.update_mu1(new_mu1)
    else:
        new_mu1 = mu1
        grad_mu1 = prev_mu1_grad

    # Calculate relative loss change for adaptive parameters
    rel_loss_change = 0
    if prev_loss is not None and curr_loss is not None:
        rel_loss_change = (curr_loss - prev_loss) / (abs(prev_loss) + 1e-10)

    grad_coefs = np.array([grad.dot(atom.calc_sig_term()) for atom in atoms]).T

    # Continue with normal atom and coef updates
    grad_Piv = np.array(
        [
            grad_coef * atom.calc_grad_ei(grad)
            for grad_coef, atom in zip(grad_coefs, atoms)
        ]
    ).T

    # Apply Nesterov-like momentum to gradient if we have previous gradient
    if prev_grad is not None:
        nesterov_grad = momentum_beta * prev_grad
        look_ahead_grad = np.array(
            [coef * atom.calc_grad_ei(grad) for coef, atom in zip(coefs, atoms)]
        ).T
        grad_Piv = nesterov_grad + (1 - momentum_beta) * look_ahead_grad

    grad_norm = np.linalg.norm(grad_Piv)

    if soft_assignment:
        # Use adaptive temperature for smoother soft assignment
        temperature = adaptive_temperature(
            ii, grad_norm, prev_temp, rel_loss_change, max_iter=algorithm.iters
        )
        new_Piv = softmax(-grad_Piv / temperature, axis=1)
    else:
        # Convert argmin to one-hot matrix
        min_indices = np.argmin(grad_Piv, axis=1)
        new_Piv = np.eye(grad_Piv.shape[1])[min_indices]

    new_E, new_c = Piv2EC(new_Piv)

    new_coefs = Coefs(new_c)
    new_atoms = deepcopy(atoms)
    for idx in range(len(atoms)):
        new_atoms[idx].e = new_E[:, idx]
        new_sig = sig - new_atoms[idx].calc_sig_term()
        new_grad = f.calc_grad(new_sig, need_L=False)
        new_atoms[idx].update_R(new_grad.Rv)
        new_atoms[idx].update_b(new_grad.pi)

    # Apply trust region constraints to limit large updates
    if prev_atoms is not None:
        # Calculate how much the atoms changed
        atom_change_magnitude = 0
        for i in range(len(atoms)):
            atom_change_magnitude += np.linalg.norm(new_atoms[i].e - prev_atoms[i].e)

        # Reduce step size if change is too large
        max_change = (
            0.2 * (1.0 - ii / algorithm.iters) + 0.02
        )  # Decreases from 0.22 to 0.02
        if atom_change_magnitude > max_change:
            scale_factor = max_change / atom_change_magnitude
            for idx in range(len(new_atoms)):
                # Scale back the change to stay within trust region
                new_atoms[idx].e = prev_atoms[idx].e + scale_factor * (
                    new_atoms[idx].e - prev_atoms[idx].e
                )

    # Calculate the update step with smoother annealing
    alpha = schedule_lr(ii)

    # Apply parameter averaging for stability (Polyak averaging)
    if ii > algorithm.iters // 2 and prev_atoms is not None:
        # Increase averaging weight as iterations progress
        avg_weight = 0.9 * (ii - algorithm.iters // 2) / (algorithm.iters // 2)
        for idx in range(len(atoms)):
            new_atoms[idx].e = (
                avg_weight * prev_atoms[idx].e + (1 - avg_weight) * new_atoms[idx].e
            )

    atoms, coefs = interpolate(atoms, coefs, new_atoms, new_coefs, alpha)
    # atoms, coefs = line_search(atoms, coefs, new_atoms, new_coefs, f)

    # Sort atoms and coefs based on np.argmax(atom.b)
    sort_indices = np.argsort([np.argmax(atom.b) for atom in atoms])
    atoms = Atoms([atoms[i] for i in sort_indices])
    coefs = Coefs([coefs[i] for i in sort_indices])

    return atoms, coefs, grad_Piv, new_mu0, grad_mu0, new_mu1, grad_mu1, temperature


def CANO(
    graph, atoms=None, coefs=None, mu0=None, mu1=None, log_results=True, leave=True
):
    n = graph.num_nodes
    num_features = graph.num_features
    c = graph.num_clusters
    ## Initialization
    if atoms is None:
        atoms, coefs = create_random_atoms(n, num_features, c, c)
    atoms_opt, coefs_opt = create_optimal_atoms(graph)

    # Initialize mu0 if not provided
    if mu0 is None:
        mu0 = data.mu0
    if mu1 is None:
        mu1 = data.mu1

    atoms, coefs = map_to_simplex(atoms, coefs)
    sig = atoms.calc_sig(coefs)
    # Initialize GraphFunc with the current mu0 value
    f = GraphFunc(graph=graph, mu0=mu0, mu1=mu1)

    opt_sig = atoms_opt.calc_sig(coefs_opt)
    f_opt, L_opt, R_opt, pi_opt = f(opt_sig, components=True)

    f_sig = f(sig)
    prev_grad = None
    prev_mu0_grad = None
    prev_mu1_grad = None
    prev_atoms = None
    prev_coefs = None
    prev_temp = None
    prev_loss = None

    # if log_results:
    bar = tqdm(total=algorithm.iters, leave=leave)
    f_old = 0
    f_values = [f_sig]
    accuracies = []
    epsilsons = []
    components = []
    y_preds = []
    Qs = []

    save_iters = np.linspace(
        0, algorithm.iters - 1, num=min(200, algorithm.iters), dtype=int
    ).tolist()

    # For exponential moving average of parameters
    ema_atoms = None
    ema_decay = 0.99  # High value for stability

    for ii in range(algorithm.iters):
        # Store previous state for adaptive parameters
        prev_atoms = deepcopy(atoms)
        prev_coefs = deepcopy(coefs)
        curr_loss = f_values[-1] if len(f_values) > 0 else None

        (
            atoms,
            coefs,
            current_grad,
            mu0,
            current_mu0_grad,
            mu1,
            current_mu1_grad,
            current_temp,
        ) = update(
            atoms,
            coefs,
            f,
            ii,
            prev_grad,
            mu0=mu0,
            prev_mu0_grad=prev_mu0_grad,
            mu1=mu1,
            prev_mu1_grad=prev_mu1_grad,
            prev_atoms=prev_atoms,
            prev_coefs=prev_coefs,
            prev_temp=prev_temp,
            prev_loss=prev_loss,
            curr_loss=curr_loss,
        )
        prev_grad = current_grad
        prev_mu0_grad = current_mu0_grad
        prev_mu1_grad = current_mu1_grad
        # prev_temp = current_temp

        # Update EMA of parameters for more stable outputs
        if ema_atoms is None:
            ema_atoms = deepcopy(atoms)
        else:
            for idx in range(len(atoms)):
                ema_atoms[idx].e = (
                    ema_decay * ema_atoms[idx].e + (1 - ema_decay) * atoms[idx].e
                )
                ema_atoms[idx].R = (
                    ema_decay * ema_atoms[idx].R + (1 - ema_decay) * atoms[idx].R
                )
                ema_atoms[idx].b = (
                    ema_decay * ema_atoms[idx].b + (1 - ema_decay) * atoms[idx].b
                )

        sig = atoms.calc_sig(coefs)
        f_sig, L_term, R_term, pi_term = f(sig, components=True)
        prev_loss = f_sig  # Store current loss for next iteration

        y_pred = np.argmax(sig.pi, axis=1)
        y_pred_test = y_pred[graph.test_mask]
        if y_pred_test.shape[0] > 0:
            acc = calc_accuracy(y_pred_test, graph.y[graph.test_mask])
        else:
            acc = 1

        results = {
            f"f": f"{f_sig:,.4f}",
            "acc": acc,
            "f_opt": f"{f_opt:,.4f}",
            # "mu0": f"{mu0:,.4f}",  # Add mu0 to results
            # "mu1": f"{mu1:,.4f}",  # Add mu1 to results
        }
        y_pred = np.argmax(sig.pi, axis=1)
        Q = np.array([np.linalg.eigh(atom.R)[1] for atom in atoms])

        # if log_results:
        bar.set_postfix(results)
        bar.update()
        f_values.append(f_sig)
        accuracies.append(acc)

        # Use EMA atoms for visualization to get smoother curves
        if ii > algorithm.iters * 0.1:  # After initial iterations
            epsilsons.append([ema_atoms.e.copy() for ema_atoms in ema_atoms])
        else:
            epsilsons.append([atoms.e.copy() for atoms in atoms])

        components.append([L_term, R_term, pi_term])

        # Change saving frequency using adaptive iterations
        if ii in save_iters:
            y_preds.append(y_pred)
            Qs.append(Q)

        # Smarter early stopping condition
        if ii > 100 and np.mean(np.abs(np.diff(f_values[-20:]))) < 1e-6:
            print(f"Early stopping at iteration {ii} due to convergence")
            break

    # For the final return, use the EMA parameters which should be smoother
    if ema_atoms is not None and ii > algorithm.iters * 0.3:
        atoms = ema_atoms

    return (
        atoms,
        coefs,
        graph,
        results,
        f_values,
        accuracies,
        epsilsons,
        components,
        (f_opt, L_opt, R_opt, pi_opt),
        (y_preds, Qs),
        # mu0_values,  # Return the mu0 values history
    )
