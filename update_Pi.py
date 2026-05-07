"""
Per-atom label distribution Pi: extension of the atom to carry a label
distribution  pi_i in Delta^{K_classes}  alongside the centroid  R_i.

Model (per the user's spec):

  - Each atom i has a centroid R_i (shape d) AND a label distribution
    pi_i in the standard simplex over K_classes.
  - For training nodes v with noisy label tilde_y_v, the predicted label
    distribution under soft assignment W is  pi_v = sum_i W_iv * pi_i,
    and the label loss is  f_label(tilde_y_v; pi_v) = -(pi_v)_{tilde_y_v}.
  - The new term added to the global objective is

        L_pi(W, Pi) = beta * sum_{v in V_train} (-(pi_v)_{tilde_y_v})
                    = -beta * sum_{v in V_train} sum_i W_iv * Pi_{i, tilde_y_v}.

Gradients:

    grad_{Pi_{i,j}} L_pi = -beta * sum_{v in V_train, tilde_y_v = j} W_iv

    grad_{W_{i,v}} L_pi  = -beta * 1[v in V_train] * Pi_{i, tilde_y_v}

Pi is constrained row-wise to the simplex.
"""

import numpy as np

from update_W import project_to_simplex

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def project_rows_to_simplex(Pi: np.ndarray) -> np.ndarray:
    """Project each row of Pi onto the standard simplex."""
    Pi = np.asarray(Pi, dtype=float)
    out = np.empty_like(Pi)
    for i in range(Pi.shape[0]):
        out[i] = project_to_simplex(Pi[i])
    return out


def _train_idx_by_class(tilde_y: np.ndarray, train_mask: np.ndarray, K_classes: int):
    """Return list of length K_classes; entry j = indices of train nodes with tilde_y=j."""
    tilde_y = np.asarray(tilde_y).astype(int)
    train_mask = np.asarray(train_mask).astype(bool)
    out = []
    for j in range(K_classes):
        out.append(np.where(train_mask & (tilde_y == j))[0])
    return out


# ----------------------------------------------------------------------
# Objective term and gradients
# ----------------------------------------------------------------------


def label_term_value(
    W: np.ndarray,
    Pi: np.ndarray,
    tilde_y: np.ndarray,
    train_mask: np.ndarray,
    beta: float,
) -> float:
    """L_pi(W, Pi) = -beta * sum_{v in train} sum_i W_iv * Pi_{i, tilde_y_v}."""
    if beta == 0.0 or Pi is None or Pi.size == 0 or W.size == 0:
        return 0.0
    tilde_y = np.asarray(tilde_y).astype(int)
    train_mask = np.asarray(train_mask).astype(bool)
    # For each v, pick column tilde_y_v of Pi  ->  Pi[:, tilde_y_v]  shape (k,)
    # Sum over train nodes of  W[:, v] . Pi[:, tilde_y_v]
    cols = Pi[:, tilde_y]  # (k, n)
    s = float(np.sum(W[:, train_mask] * cols[:, train_mask]))
    return -beta * s


def label_term_subgradient_Pi(
    W: np.ndarray,
    tilde_y: np.ndarray,
    train_mask: np.ndarray,
    beta: float,
    K_classes: int,
) -> np.ndarray:
    """
    grad_{Pi_{i,j}} L_pi = -beta * sum_{v in train, tilde_y_v=j} W_iv.
    Returns a (k, K_classes) array.
    """
    k = W.shape[0]
    H = np.zeros((k, K_classes), dtype=float)
    if beta == 0.0:
        return H
    idx_by_class = _train_idx_by_class(tilde_y, train_mask, K_classes)
    for j, idx in enumerate(idx_by_class):
        if len(idx) > 0:
            H[:, j] = -beta * W[:, idx].sum(axis=1)
    return H


# ----------------------------------------------------------------------
# Pi update
# ----------------------------------------------------------------------


def update_Pi_closed_form(
    W: np.ndarray,
    tilde_y: np.ndarray,
    train_mask: np.ndarray,
    beta: float,
    K_classes: int,
) -> np.ndarray:
    """
    Closed-form minimizer of L_pi over rows of Pi (each row in simplex).

    Since L_pi is linear in Pi, the row-wise minimum is at the vertex
    e_{j*(i)} where  j*(i) = argmax_j sum_{v in train, tilde_y_v=j} W_iv.

    Returns Pi of shape (k, K_classes); rows are one-hot.  Rows where the
    max is zero (no signal) are set to uniform.
    """
    k = W.shape[0]
    Pi = np.zeros((k, K_classes), dtype=float)
    if beta == 0.0 or k == 0:
        Pi[:] = 1.0 / max(K_classes, 1)
        return Pi
    idx_by_class = _train_idx_by_class(tilde_y, train_mask, K_classes)
    score = np.zeros((k, K_classes), dtype=float)
    for j, idx in enumerate(idx_by_class):
        if len(idx) > 0:
            score[:, j] = W[:, idx].sum(axis=1)
    rowmax = score.max(axis=1)
    for i in range(k):
        if rowmax[i] <= 0.0:
            Pi[i, :] = 1.0 / K_classes
        else:
            j_star = int(np.argmax(score[i]))
            Pi[i, j_star] = 1.0
    return Pi


def update_Pi_projected_gradient(
    Pi_init: np.ndarray,
    W: np.ndarray,
    tilde_y: np.ndarray,
    train_mask: np.ndarray,
    beta: float,
    step_size: float = 1.0,
    max_iter: int = 20,
    tol: float = 1e-8,
) -> np.ndarray:
    """
    Projected-gradient minimization of L_pi over rows of Pi (simplex).
    With a linear loss and a unit step on a normalized score, this
    typically converges in 1-2 iterations to the vertex.
    """
    Pi = np.asarray(Pi_init, dtype=float).copy()
    K_classes = Pi.shape[1]
    if beta == 0.0:
        return project_rows_to_simplex(Pi)
    for _ in range(max_iter):
        H = label_term_subgradient_Pi(W, tilde_y, train_mask, beta, K_classes)
        # Normalize step by row L_inf to keep updates sane regardless of |train|
        scale = np.maximum(np.abs(H).max(axis=1, keepdims=True), 1e-12)
        Pi_new = project_rows_to_simplex(Pi - step_size * H / scale)
        change = float(np.linalg.norm(Pi_new - Pi))
        Pi = Pi_new
        if change < tol:
            break
    return Pi


def update_Pi(
    Pi_init: np.ndarray,
    W: np.ndarray,
    tilde_y: np.ndarray,
    train_mask: np.ndarray,
    beta: float,
    K_classes: int,
    method: str = "cf",
    step_size: float = 1.0,
    max_iter: int = 20,
    tol: float = 1e-8,
) -> np.ndarray:
    """Unified Pi update.

    method:
      - "cf"  : closed-form vertex (argmax row score).
      - "pg"  : projected-gradient on the simplex.
    """
    method = method.lower()
    if method in ("cf", "closed_form", "closed-form"):
        return update_Pi_closed_form(W, tilde_y, train_mask, beta, K_classes)
    if method == "pg":
        return update_Pi_projected_gradient(
            Pi_init,
            W,
            tilde_y,
            train_mask,
            beta,
            step_size=step_size,
            max_iter=max_iter,
            tol=tol,
        )
    raise ValueError("Unknown Pi update method: " + method)


# ----------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------


def node_label_distribution(W: np.ndarray, Pi: np.ndarray) -> np.ndarray:
    """
    Per-node predicted label distribution

        pi_v = sum_i lambda_i * e_{iv}^2 * pi_i  =  sum_i W_{iv} * pi_i

    (since lambda_i = sum_v W_{iv} and  e_{iv}^2 = W_{iv} / lambda_i  imply
     lambda_i * e_{iv}^2 = W_{iv}).

    Returns an (n, K_classes) array.  Rows are NOT renormalized; columns of W
    summing to 1 already guarantee each row of the output is a valid simplex
    point (convex combination of simplex rows of Pi).
    """
    if W.size == 0 or Pi.size == 0:
        return np.zeros((W.shape[1], Pi.shape[1] if Pi.ndim == 2 else 0), dtype=float)
    return W.T @ Pi  # (n, k) @ (k, K_classes) -> (n, K_classes)


def predict_node_labels(W: np.ndarray, Pi: np.ndarray) -> np.ndarray:
    """
    Hard label prediction  hat_y_v = argmax_j (pi_v)_j  for every node v.
    Use this on V_test (or any node set) after fitting.
    """
    return node_label_distribution(W, Pi).argmax(axis=1)
