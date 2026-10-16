"""
Node losses f_v for the unified problem (14).

The solver touches the node information only through the `NodeLoss` interface
defined here, so it is not tied to a particular loss.  Two losses are provided:

* `SquaredLoss`     f_v(theta) = 1/2 ||x_v - theta||^2             (convex clustering, eq. (3))
* `LabelAwareLoss`  f_v(a, pi) = 1/2 ||x_v - a||^2 - nu 1{v in train} pi_{y_v}   (eq. (22))

A new loss only has to implement `values`, `pointwise`, `grad_atoms` and set
`smoothness` (and `project` if Theta_f is not the whole space); every other
operation has a generic first-order fallback.  The Frank-Wolfe enhancement step
additionally needs `lmo` (a linear minimization oracle over a compact Theta_f).

Notation used in the docstrings:
    n      number of nodes,  k  number of atoms,  dim  dimension of theta
    Theta  (k, dim) array, row i = atom parameter theta_i
    Omega  (n, k) array of node-atom weights, rows in the simplex
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist

# ============================================================
# Simplex helpers
# ============================================================


def project_rows_simplex(V: np.ndarray) -> np.ndarray:
    """Euclidean projection of every row of V onto the probability simplex
    (sort-based algorithm; O(k log k) per row)."""
    V = np.atleast_2d(np.asarray(V, dtype=float))
    n, k = V.shape
    U = -np.sort(-V, axis=1)
    css = np.cumsum(U, axis=1) - 1.0
    cond = U - css / np.arange(1, k + 1) > 0
    r = k - 1 - np.argmax(cond[:, ::-1], axis=1)  # last index where cond holds
    tau = css[np.arange(n), r] / (r + 1)
    return np.maximum(V - tau[:, None], 0.0)


def simplex_argmin_linear(C: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """Row-wise minimizer of <c, pi> over the simplex.

    The minimum of a linear function over the simplex is attained at a vertex.
    If several coordinates tie for the minimum, the mass is shared uniformly
    among them (barycenter of the minimizing face), so no class is favoured."""
    C = np.atleast_2d(np.asarray(C, dtype=float))
    m = C.min(axis=1, keepdims=True)
    scale = np.maximum(1.0, np.abs(C).max(axis=1, keepdims=True))
    mask = C <= m + tol * scale
    return mask / mask.sum(axis=1, keepdims=True)


# ============================================================
# Interface
# ============================================================


class NodeLoss:
    """Convex node losses f_v : R^dim -> (-inf, +inf], evaluated at atoms.

    Required (no default):
        values(Theta)            -> (n, k)   entries f_v(theta_i)
        pointwise(Theta_rows)    -> (n,)     entries f_v(theta_v), one parameter per node
        grad_atoms(Theta, Wt)    -> (k, dim) row i = sum_v Wt[v, i] grad f_v(theta_i)
        smoothness               Lipschitz constant of grad f_v (used for step sizes)

    Optional (generic fallbacks below):
        project(Theta)           Euclidean projection of every row onto Theta_f
        decode(Theta)            the map phi: parameter -> predicted label
        lmo(Grad)                argmin_{theta in Theta_f} <Grad_i, theta> (Frank-Wolfe only)
        weighted_argmin(w, c)    argmin_theta sum_v w_v f_v(theta) + <c, theta>   (eq. (121))
        per_node_argmin(s, C)    row v: argmin_theta s f_v(theta) + <C_v, theta>  (eq. (117))
        minimize_quadratic_surrogate(...)   Theta-step of the default enhancement
    """

    n: int
    dim: int
    smoothness: float = 1.0

    # ---- required -------------------------------------------------------
    def values(self, Theta: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def pointwise(self, Theta_rows: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def grad_atoms(self, Theta: np.ndarray, Wt: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    # ---- optional -------------------------------------------------------
    def project(self, Theta: np.ndarray) -> np.ndarray:
        return Theta

    def decode(self, Theta: np.ndarray):
        return None

    def lmo(self, Grad: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "the Frank-Wolfe enhancement step needs a compact Theta_f (implement lmo)"
        )

    def weighted_argmin(self, w, c, theta0=None, iters: int = 300):
        """argmin_{theta in Theta_f} sum_v w_v f_v(theta) + <c, theta>.

        Generic fallback: accelerated projected gradient (FISTA) with step
        1 / (smoothness * sum(w)).  Needs f_v coercive or Theta_f compact."""
        w = np.asarray(w, dtype=float)
        x = np.zeros(self.dim) if theta0 is None else np.array(theta0, dtype=float)
        x = self.project(x[None])[0]
        lip = self.smoothness * max(w.sum(), 1e-12)
        y, t = x.copy(), 1.0
        for _ in range(iters):
            g = self.grad_atoms(y[None], w[:, None])[0] + c
            xn = self.project((y - g / lip)[None])[0]
            tn = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
            y = xn + ((t - 1.0) / tn) * (xn - x)
            if np.linalg.norm(xn - x) <= 1e-10 * max(1.0, np.linalg.norm(x)):
                x = xn
                break
            x, t = xn, tn
        return x

    def per_node_argmin(self, scale: float, C: np.ndarray) -> np.ndarray:
        """Row v: argmin_theta scale * f_v(theta) + <C_v, theta>  (one call per node)."""
        out = np.empty((self.n, self.dim))
        for v in range(self.n):
            w = np.zeros(self.n)
            w[v] = scale
            out[v] = self.weighted_argmin(w, C[v])
        return out

    def minimize_quadratic_surrogate(self, Omega, Q, wf, ws, Theta0, iters: int = 200):
        """argmin_Theta  wf * sum_{v,i} Omega[v,i] f_v(theta_i) + (ws/2) tr(Theta^T Q Theta)
        over Theta_f^k.  This is one majorize-minimize step for the fusion term
        (see `CoGEnT._theta_step`).  Generic fallback: FISTA with the exact
        Lipschitz constant; returns Theta0 if the surrogate did not decrease."""

        def surr(Th):
            return wf * float(np.sum(Omega * self.values(Th))) + 0.5 * ws * float(
                np.sum(Th * (Q @ Th))
            )

        lq = float(np.linalg.eigvalsh(Q).max()) if ws > 0 else 0.0
        lip = wf * self.smoothness * float(Omega.sum(0).max()) + ws * lq + 1e-12
        X = Theta0.copy()
        Y, t = X.copy(), 1.0
        for _ in range(iters):
            g = wf * self.grad_atoms(Y, Omega) + ws * (Q @ Y)
            Xn = self.project(Y - g / lip)
            tn = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
            Y = Xn + ((t - 1.0) / tn) * (Xn - X)
            X, t = Xn, tn
        return X if surr(X) <= surr(Theta0) else Theta0


# ============================================================
# Squared loss (convex clustering)
# ============================================================


class SquaredLoss(NodeLoss):
    """f_v(theta) = 1/2 ||x_v - theta||^2,  Theta_f = R^d  (problem (3) of the paper).

    All oracles have closed forms because the loss is an isotropic quadratic."""

    smoothness = 1.0

    def __init__(self, X: np.ndarray):
        self.X = np.asarray(X, dtype=float)
        self.n, self.dim = self.X.shape

    def values(self, Theta):
        return 0.5 * cdist(self.X, np.atleast_2d(Theta), "sqeuclidean")

    def pointwise(self, Theta_rows):
        return 0.5 * np.sum((self.X - Theta_rows) ** 2, axis=1)

    def grad_atoms(self, Theta, Wt):
        # sum_v Wt[v,i] (theta_i - x_v) = (sum_v Wt[v,i]) theta_i - (Wt^T X)_i
        return Wt.sum(0)[:, None] * Theta - Wt.T @ self.X

    def weighted_argmin(self, w, c, theta0=None, iters=0):
        # weighted mean shifted by the linear term
        return (w @ self.X - c) / max(float(np.sum(w)), 1e-12)

    def per_node_argmin(self, scale, C):
        return self.X - C / scale

    def minimize_quadratic_surrogate(self, Omega, Q, wf, ws, Theta0, iters=0):
        # stationarity: (wf diag(m) + ws Q) Theta = wf Omega^T X,  m_i = sum_v Omega[v,i]
        H = wf * np.diag(Omega.sum(0)) + ws * Q
        H += 1e-12 * max(1.0, np.trace(H)) * np.eye(H.shape[0])
        return np.linalg.solve(H, wf * (Omega.T @ self.X))


# ============================================================
# Label-aware loss (Sec. 6, eq. (22))
# ============================================================


class LabelAwareLoss(NodeLoss):
    """Label-aware node loss of Sec. 6, eq. (22):

        f_v(a, pi) = 1/2 ||x_v - a||^2 - nu * 1{v in V_train} * pi_{y_v},
        theta = (a, pi),   a in R^d (or the ball ||a|| <= a_radius),   pi in the simplex,
        phi(theta) = argmax_c pi_c   (the predicted label).

    The feature block `a` is a centroid; the label block `pi` is a distribution over
    the K classes.  Training nodes reward atoms whose pi puts mass on their observed
    label; test nodes only contribute the feature term.

    Parameters
    ----------
    X : (n, d) features;  y_obs : (n,) labels (only entries with train_mask are used)
    train_mask : (n,) bool;  n_classes : K;  nu : weight of the label term
    a_radius : optional bound ||a|| <= a_radius.  Needed by the Frank-Wolfe
        enhancement step: its atom LMO minimizes a linear function over Theta_f,
        which is unbounded on R^d.  Use the same value in every run you compare.
    """

    smoothness = 1.0

    def __init__(self, X, y_obs, train_mask, n_classes: int, nu: float = 1.0, a_radius=None):
        self.X = np.asarray(X, dtype=float)
        self.n, self.d = self.X.shape
        self.K = int(n_classes)
        self.dim = self.d + self.K
        self.nu = float(nu)
        self.a_radius = None if a_radius is None else float(a_radius)
        self.train = np.asarray(train_mask, dtype=bool)
        y = np.asarray(y_obs, dtype=int)
        self.Y = np.zeros((self.n, self.K))
        self.Y[self.train, y[self.train]] = 1.0  # one-hot rows for training nodes, zero otherwise

    def _split(self, Theta):
        Theta = np.atleast_2d(Theta)
        return Theta[:, : self.d], Theta[:, self.d :]

    def values(self, Theta):
        A, P = self._split(Theta)
        return 0.5 * cdist(self.X, A, "sqeuclidean") - self.nu * (self.Y @ P.T)

    def pointwise(self, Theta_rows):
        A, P = self._split(Theta_rows)
        return 0.5 * np.sum((self.X - A) ** 2, axis=1) - self.nu * np.sum(self.Y * P, 1)

    def grad_atoms(self, Theta, Wt):
        A, _ = self._split(Theta)
        ga = Wt.sum(0)[:, None] * A - Wt.T @ self.X  # feature block
        gp = -self.nu * (Wt.T @ self.Y)  # label block (linear in pi)
        return np.hstack([ga, gp])

    def _ball(self, A):
        """Projection of every row of A onto {||a|| <= a_radius} (identity if unbounded)."""
        if self.a_radius is None:
            return A
        nrm = np.linalg.norm(A, axis=1, keepdims=True)
        return A * np.minimum(1.0, self.a_radius / np.maximum(nrm, 1e-12))

    def project(self, Theta):
        A, P = self._split(Theta)
        return np.hstack([self._ball(A), project_rows_simplex(P)])

    def lmo(self, Grad):
        """Atom LMO (112) over Theta_f = ball x simplex:
        a = -R g_a / ||g_a||,  pi = vertex of the simplex with the smallest g_pi."""
        if self.a_radius is None:
            raise NotImplementedError(
                "Frank-Wolfe atom step: the LMO over a in R^d is unbounded; set a_radius"
            )
        Ga, Gp = self._split(Grad)
        nrm = np.linalg.norm(Ga, axis=1, keepdims=True)
        A = -self.a_radius * Ga / np.maximum(nrm, 1e-12)
        return np.hstack([A, simplex_argmin_linear(Gp)])

    # The feature part of the two problems below is an isotropic quadratic, so
    # projecting the unconstrained minimizer onto the ball gives the constrained one.
    def weighted_argmin(self, w, c, theta0=None, iters=0):
        a = (w @ self.X - c[: self.d]) / max(float(np.sum(w)), 1e-12)
        p = simplex_argmin_linear(-self.nu * (w @ self.Y) + c[self.d :])[0]
        return np.concatenate([self._ball(a[None])[0], p])

    def per_node_argmin(self, scale, C):
        A = self.X - C[:, : self.d] / scale
        P = simplex_argmin_linear(-scale * self.nu * self.Y + C[:, self.d :])
        return np.hstack([self._ball(A), P])

    def minimize_quadratic_surrogate(self, Omega, Q, wf, ws, Theta0, iters=40):
        """Majorize-minimize step for (22): the surrogate separates into
        * feature block: a k x k linear system  (wf diag(m) + ws Q) A = wf Omega^T X,
        * label block:   min_P -wf nu <B, P> + (ws/2) tr(P^T Q P),  rows of P in the simplex,
          with B = Omega^T Y, solved by FISTA with step 1/(ws lambda_max(Q)).
        (The caller projects the result onto Theta_f and accepts it only if the
        true objective decreases.)"""
        k = Omega.shape[1]
        H = wf * np.diag(Omega.sum(0)) + ws * Q
        H += 1e-12 * max(1.0, np.trace(H)) * np.eye(k)
        A = np.linalg.solve(H, wf * (Omega.T @ self.X))
        B = Omega.T @ self.Y
        P0 = Theta0[:, self.d :]
        lq = float(np.linalg.eigvalsh(Q).max()) if ws > 0 else 0.0
        if ws * lq <= 1e-12:
            P = simplex_argmin_linear(-wf * self.nu * B)
        else:

            def surr_p(P):
                return -wf * self.nu * float(np.sum(B * P)) + 0.5 * ws * float(
                    np.sum(P * (Q @ P))
                )

            X, Y, t = P0.copy(), P0.copy(), 1.0
            for _ in range(iters):
                g = -wf * self.nu * B + ws * (Q @ Y)
                Xn = project_rows_simplex(Y - g / (ws * lq))
                tn = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
                Y = Xn + ((t - 1.0) / tn) * (Xn - X)
                small = np.abs(Xn - X).max() <= 1e-7
                X, t = Xn, tn
                if small:
                    break
            P = X if surr_p(X) <= surr_p(P0) else P0
        return np.hstack([A, P])

    def decode(self, Theta, seed: int = 0):
        """phi(theta) = argmax_c pi_c for every row; ties are broken uniformly at
        random with a fixed seed (so results are reproducible)."""
        _, P = self._split(Theta)
        rng = np.random.default_rng(seed)
        top = P.max(axis=1, keepdims=True)
        pred = np.empty(P.shape[0], dtype=int)
        for i, row in enumerate(P >= top - 1e-9):
            pred[i] = rng.choice(np.flatnonzero(row))
        return pred
