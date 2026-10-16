"""
CoGEnT-style atomic conditional-gradient solver for the unified problem (14)
(App. F / Algorithm 1 of the paper).  A full derivation of every step is in
docs/solver_explained.md; equation numbers below refer to the paper.

Objective
---------
The solver minimizes the reduced objective (104), divided by n*mu:

    Phi(Omega, Theta) =  wg * ( - sum_{u != v} Abar0_uv L_uv )          graph term
                       + wf * sum_{v,i} Omega[v,i] f_v(theta_i)          lifted node loss
                       + ws * sum_{u<v} || theta_u - theta_v ||_2        fusion (sum of norms)

    Omega  (n, k)  node-atom weights, every row in the simplex  (omega_{v,i} = lambda_i e_{i,v}^2)
    Theta  (k, p)  atom parameters theta_i in Theta_f
    L = E E^T with E = sqrt(Omega)    (so L_vv = 1 and 0 <= L_uv <= 1)
    theta_v = sum_i Omega[v,i] theta_i   (node parameter; the matrix T = Omega Theta)
    Abar0   polarized adjacency: +1 edge, -1 non-edge, 0 on the diagonal

    (wg, wf, ws) = (1/(n mu), 1, gamma/n)   0 < mu < inf   combined
                 = (0,        1, gamma/n)   mu = inf       node information only
                 = (1,        0, 0)         mu = 0         graph only (Theta decoded at the end)

Algorithm (one outer iteration)
-------------------------------
1. Forward step (F.4): an oracle proposes new atoms (S, theta) -- the nodes in S
   move part of their mass to a new atom with parameter theta -- ranked by the
   first-order rate (115)-(116); the best candidates are line-searched on the
   exact Phi and the best one is added if Phi decreases.
2. Enhancement step (F.3), with the active set fixed.  Two options:
     enhancement="bcd"  (default) block-coordinate descent: every row of Omega
                        is replaced by the best of a few candidate rows (exact
                        change of Phi), then a majorize-minimize step on Theta;
     enhancement="fw"   Frank-Wolfe exactly as in F.3: a row LMO step on Omega
                        (105), (108)-(111) and an atom LMO step on Theta (106),
                        (112)-(113), each with a line search on Phi.
3. Backward step (F.5): CoGEnT truncation rule (123).
Every step either lowers Phi or leaves it unchanged.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform

from .losses import NodeLoss

try:  # optional numba acceleration of the default row sweep (same results)
    from ._numba_kernels import omega_sweep as _omega_sweep_nb
except Exception:  # numba not installed
    _omega_sweep_nb = None


# ============================================================
# Fusion (sum-of-norms) helpers
# ============================================================


def _son_value(T: np.ndarray) -> float:
    """sum_{u<v} ||T_u - T_v||."""
    return float(pdist(T).sum()) if T.shape[0] > 1 else 0.0


def _son_subgrad(T: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """G_v = sum_{u != v} g_vu with g_vu = (T_v - T_u)/||T_v - T_u||, and g_vu = 0
    for tied pairs (T_v = T_u)."""
    diff = T[:, None, :] - T[None, :, :]
    nrm = np.linalg.norm(diff, axis=2)
    inv = np.where(nrm > eps, 1.0 / np.maximum(nrm, eps), 0.0)
    return np.einsum("uvd,uv->ud", diff, inv)


# ============================================================
# Solver
# ============================================================


class CoGEnT:
    """Atomic conditional-gradient solver for problem (14) (Algorithm 1).

    Parameters
    ----------
    loss : NodeLoss            node losses f_v (see cogent/losses.py)
    A_graph : (n, n) 0/1 adjacency, or None (then node only)
    gamma, mu : the paper's gamma and mu; mu = inf gives node only, mu = 0 graph only
    max_outer : outer iterations T (default 2n)
    inner_cycles : enhancement cycles per outer iteration
    eta : truncation parameter of rule (123)
    max_atoms : cap on the number of atoms (default n, i.e. no cap; when rho > gamma/K
        the node-only optimum splits classes into many atoms)
    lmo_alternations, n_eig : oracle search effort (Sec. 4.4 of docs/solver_explained.md)
    n_line_search : number of oracle candidates evaluated by the exact line search
    backward_cycles : enhancement cycles used to re-optimize a deletion candidate
    n_vertex_cands : full-move candidates per row in the default row sweep
    enhancement : "bcd" (default) or "fw" (Frank-Wolfe as written in App. F.3)
    fw_eps : safeguard max(e_{i,v}, fw_eps) in the partial derivative (105) ("fw" only)
    fw_batch : vectorized line search on a fixed step grid ("fw" only); if False,
        a sequential grid (fw_ls_grid points) + golden search (fw_ls_golden steps)
    seed : seeds the random row order of the default row sweep
    use_numba : use the numba kernel for the default row sweep if available
    """

    def __init__(
        self,
        loss: NodeLoss,
        A_graph: np.ndarray | None,
        gamma: float,
        mu: float,
        max_outer: int | None = None,
        inner_cycles: int = 3,
        theta_iters: int = 5,
        eta: float = 0.5,
        max_atoms: int | None = None,
        lmo_alternations: int = 4,
        n_eig: int = 3,
        backward_cycles: int = 1,
        n_vertex_cands: int = 6,
        n_line_search: int = 6,
        enhancement: str = "bcd",
        fw_eps: float = 1e-8,
        fw_ls_grid: int = 10,
        fw_ls_golden: int = 10,
        fw_batch: bool = True,
        seed: int = 0,
        verbose: bool = False,
        use_numba: bool = True,
    ):
        self.loss = loss
        self.n = loss.n
        self.p = loss.dim
        self.gamma = float(gamma)
        self.mu = float(mu)
        n = self.n
        # weights of the three terms of Phi (objective (104) divided by n*mu)
        if A_graph is None or np.isinf(self.mu):
            self.wg, self.wf, self.ws = 0.0, 1.0, self.gamma / n  # node only
            self.Abar0 = np.zeros((n, n))
        elif self.mu == 0.0:
            self.wg, self.wf, self.ws = 1.0, 0.0, 0.0  # graph only
        else:
            self.wg, self.wf, self.ws = 1.0 / (n * self.mu), 1.0, self.gamma / n  # combined
        if A_graph is not None and self.wg > 0:
            Ab = 2.0 * np.asarray(A_graph, dtype=float) - 1.0  # +1 edge, -1 non-edge
            np.fill_diagonal(Ab, 0.0)  # the diagonal only adds a constant (L_vv = 1)
            self.Abar0 = Ab
        self.max_outer = 2 * n if max_outer is None else int(max_outer)
        self.inner_cycles = int(inner_cycles)
        self.theta_iters = int(theta_iters)
        self.eta = float(eta)
        self.max_atoms = n if max_atoms is None else int(max_atoms)
        self.lmo_alternations = int(lmo_alternations)
        self.n_eig = int(n_eig)
        self.backward_cycles = int(backward_cycles)
        self.n_vertex_cands = int(n_vertex_cands)
        self.n_line_search = int(n_line_search)
        if enhancement not in ("bcd", "fw"):
            raise ValueError("enhancement must be 'bcd' or 'fw'")
        self.enhancement = enhancement
        self.fw_eps = float(fw_eps)
        self.fw_ls = (int(fw_ls_grid), int(fw_ls_golden))
        self.fw_batch = bool(fw_batch)
        self.rng = np.random.default_rng(seed)
        self.verbose = verbose
        self.use_numba = bool(use_numba) and _omega_sweep_nb is not None

    # ------------------------------------------------------------------
    # State and objective
    # ------------------------------------------------------------------
    def _set_state(self, Om: np.ndarray, Th: np.ndarray):
        """Set (Omega, Theta): rows of Omega renormalized, Theta projected onto Theta_f."""
        self.Om = np.maximum(Om, 0.0)
        self.Om /= self.Om.sum(1, keepdims=True)
        self.Th = self.loss.project(np.atleast_2d(Th).copy())
        self._refresh()

    def _refresh(self):
        """Recompute the caches E = sqrt(Omega), F = f_v(theta_i), T = Omega Theta, Abar0 E."""
        c = np.ascontiguousarray
        self.Om = c(self.Om, dtype=float)
        self.Th = c(self.Th, dtype=float)
        self.E = np.sqrt(self.Om)
        self.F = c(self.loss.values(self.Th)) if self.wf > 0 else np.zeros_like(self.Om)
        self.T = c(self.Om @ self.Th)
        self.AE = c(self.Abar0 @ self.E) if self.wg > 0 else np.zeros_like(self.Om)

    def _snapshot(self):
        return (self.Om.copy(), self.Th.copy())

    def _restore(self, snap):
        self.Om, self.Th = snap[0].copy(), snap[1].copy()
        self._refresh()

    def objective(self) -> float:
        """Phi at the current point (graph term uses sum Abar0 L = <E, Abar0 E>)."""
        val = 0.0
        if self.wg > 0:
            val -= self.wg * float(np.sum(self.E * self.AE))
        if self.wf > 0:
            val += self.wf * float(np.sum(self.Om * self.F))
        if self.ws > 0:
            val += self.ws * _son_value(self.T)
        return val

    def _drop_empty_atoms(self, tol: float = 1e-12):
        """Remove atoms that no node uses (does not change Phi up to `tol`)."""
        keep = self.Om.sum(0) > tol
        if keep.all() or not keep.any():
            return
        self.Om = self.Om[:, keep]
        self.Om /= self.Om.sum(1, keepdims=True)
        self.Th = self.Th[keep]
        self._refresh()

    def _inner(self, cycles: int):
        """Enhancement step (F.3): `cycles` rounds of [rows of Omega; drop empty atoms;
        atoms Theta], stopping early when a round gains less than 1e-9 (relative)."""
        for _ in range(cycles):
            before = self.objective()
            if self.enhancement == "fw":
                self._fw_omega_step()
                self._drop_empty_atoms()
                self._fw_theta_step()
            else:
                self._omega_sweep()
                self._drop_empty_atoms()
                self._theta_step()
            after = self.objective()
            if before - after <= 1e-9 * max(1.0, abs(before)):
                break

    # ------------------------------------------------------------------
    # Enhancement option "bcd" (default): exact row moves + MM on Theta
    # (docs/solver_explained.md, Sec. 5.1-5.2)
    # ------------------------------------------------------------------
    def _omega_sweep(self) -> float:
        """One pass of block-coordinate descent over the rows of Omega.

        For row v (random order) the candidate rows are
          * full moves to an atom (simplex vertices): the n_vertex_cands atoms with the
            smallest partial derivative (105), plus the atoms the node already uses;
          * Frank-Wolfe points (1-a) w_v + a e_{i*}, a in {0.25, 0.5}, i* = argmin of (105);
          * pairwise points: move 50% or 100% of the mass of the worst used atom to i*.
        The exact change of Phi is computed for every candidate (only row/column v of L
        and theta_v change); the best is applied only if it lowers Phi.
        The numba kernel (_numba_kernels.py) does exactly the same."""
        n, k = self.Om.shape
        if k == 1:
            return 0.0
        wg, wf, ws = self.wg, self.wf, self.ws
        if self.use_numba:
            return _omega_sweep_nb(
                self.Om, self.E, self.Th, self.T, self.AE, self.F, self.Abar0,
                wg, wf, ws, self.rng.permutation(n), self.n_vertex_cands,
            )
        eye = np.eye(k)
        total = 0.0
        for v in self.rng.permutation(n):
            w_v, e_v = self.Om[v], self.E[v]
            r_v = self.AE[v]  # (Abar0 E)_v
            F_v = self.F[v]
            if ws > 0:
                dlt = self.T[v] - self.T
                dist_v = np.linalg.norm(dlt, axis=1)
                dist_v[v] = 0.0
                inv = np.where(dist_v > 1e-12, 1.0 / np.maximum(dist_v, 1e-12), 0.0)
                G_v = inv @ dlt  # fusion subgradient of node v
            # partial derivatives (105): only used to choose which candidates to try
            D = wf * F_v
            if wg > 0:
                D = D - wg * r_v / np.maximum(e_v, 1e-8)
            if ws > 0:
                D = D + ws * (self.Th @ G_v)
            i_to = int(np.argmin(D))
            support = np.flatnonzero(w_v > 1e-12)
            j_away = int(support[np.argmax(D[support])])
            if k > self.n_vertex_cands:
                verts = np.union1d(np.argpartition(D, self.n_vertex_cands)[: self.n_vertex_cands], support)
                cands = [eye[verts]]
            else:
                cands = [eye]
            fw = np.array([0.25, 0.5])[:, None] * (eye[i_to] - w_v) + w_v
            cands.append(fw)
            if j_away != i_to:
                amt = np.array([0.5, 1.0])[:, None] * w_v[j_away]
                pw = np.repeat(w_v[None], 2, 0)
                pw[:, j_away] -= amt[:, 0]
                pw[:, i_to] += amt[:, 0]
                cands.append(pw)
            Wc = np.maximum(np.vstack(cands), 0.0)
            Wc /= Wc.sum(1, keepdims=True)
            Ec = np.sqrt(Wc)
            # exact change of Phi for every candidate row
            delta = np.zeros(Wc.shape[0])
            if wg > 0:
                delta -= 2.0 * wg * ((Ec - e_v) @ r_v)
            if wf > 0:
                delta += wf * ((Wc - w_v) @ F_v)
            if ws > 0:
                Tc = Wc @ self.Th
                Dc = cdist(Tc, self.T)
                Dc[:, v] = 0.0
                delta += ws * (Dc.sum(1) - dist_v.sum())
            b = int(np.argmin(delta))
            if delta[b] < -1e-12:
                total += float(delta[b])
                if wg > 0:  # incremental update of the cache Abar0 E
                    self.AE += np.outer(self.Abar0[:, v], Ec[b] - e_v)
                self.Om[v] = Wc[b]
                self.E[v] = Ec[b]
                self.T[v] = Wc[b] @ self.Th
        return total

    def _theta_obj(self, Th) -> float:
        """The part of Phi that depends on Theta (graph term excluded)."""
        val = 0.0
        if self.wf > 0:
            val += self.wf * float(np.sum(self.Om * self.loss.values(Th)))
        if self.ws > 0:
            val += self.ws * _son_value(self.Om @ Th)
        return val

    def _theta_step(self, iters: int | None = None, wf=None, ws=None):
        """Majorize-minimize on Theta with Omega fixed (a convex problem).

        Every fusion pair is majorized with ||z|| <= ||z||^2/(2c) + c/2,
        c = max(||z0||, 1e-6 * scale) at the current point; pairs with identical rows
        of Omega are skipped (their difference is 0 for every Theta).  The surrogate
        wf sum Omega f + (ws/2) tr(Theta^T Q Theta), Q = Omega^T L_W Omega, is minimized
        by `loss.minimize_quadratic_surrogate`; the new Theta is accepted only if the
        true Theta-objective decreases.  `wf`, `ws` override the weights (used to
        decode Theta in the graph-only regime)."""
        wf = self.wf if wf is None else wf
        ws = self.ws if ws is None else ws
        if wf == 0:
            return
        iters = self.theta_iters if iters is None else iters
        Om = self.Om
        same = squareform(pdist(Om)) <= 1e-12  # pairs whose difference is identically 0
        saved = (self.wf, self.ws)
        self.wf, self.ws = wf, ws
        cur = self._theta_obj(self.Th)
        for _ in range(iters):
            k = Om.shape[1]
            if ws > 0:
                T = Om @ self.Th
                Dm = squareform(pdist(T))
                scale = max(float(Dm.max()), 1.0)
                Wp = np.where(same, 0.0, 1.0 / np.maximum(Dm, 1e-6 * scale))
                np.fill_diagonal(Wp, 0.0)
                Lw = np.diag(Wp.sum(1)) - Wp  # weighted graph Laplacian
                Q = Om.T @ Lw @ Om
                Q = 0.5 * (Q + Q.T)
            else:
                Q = np.zeros((k, k))
            Th_new = self.loss.project(
                self.loss.minimize_quadratic_surrogate(Om, Q, wf, ws, self.Th)
            )
            new = self._theta_obj(Th_new)
            if new < cur - 1e-12 * max(1.0, abs(cur)):
                done = cur - new <= 1e-9 * max(1.0, abs(cur))
                self.Th, cur = Th_new, new
                if done:
                    break
            else:
                break
        self.wf, self.ws = saved
        self._refresh()

    # ------------------------------------------------------------------
    # Enhancement option "fw": Frank-Wolfe as written in App. F.3
    # (docs/solver_explained.md, Sec. 5.3)
    # ------------------------------------------------------------------
    def _objective_at(self, Om, Th, F=None, wf=None, ws=None, graph=True):
        """Phi at an arbitrary (Omega, Theta) without touching the state."""
        wf = self.wf if wf is None else wf
        ws = self.ws if ws is None else ws
        val = 0.0
        if graph and self.wg > 0:
            E = np.sqrt(np.maximum(Om, 0.0))
            val -= self.wg * float(np.sum(E * (self.Abar0 @ E)))
        if wf > 0:
            val += wf * float(np.sum(Om * (self.loss.values(Th) if F is None else F)))
        if ws > 0:
            val += ws * _son_value(Om @ Th)
        return val

    @staticmethod
    def _line_search_unit(f, n_grid: int = 10, n_golden: int = 10):
        """Sequential line search on (0, 1]: equally spaced grid, then golden section
        around the best grid point (used when fw_batch=False)."""
        grid = np.linspace(0.0, 1.0, n_grid + 1)[1:]
        vals = [f(a) for a in grid]
        j = int(np.argmin(vals))
        a_best, v_best = grid[j], vals[j]
        lo = grid[j - 1] if j > 0 else 0.0
        hi = grid[min(j + 1, n_grid - 1)]
        gr = 0.5 * (np.sqrt(5.0) - 1.0)
        x1, x2 = hi - gr * (hi - lo), lo + gr * (hi - lo)
        f1, f2 = f(x1), f(x2)
        for _ in range(n_golden):
            if f1 < f2:
                hi, x2, f2 = x2, x1, f1
                x1 = hi - gr * (hi - lo)
                f1 = f(x1)
            else:
                lo, x1, f1 = x1, x2, f2
                x2 = lo + gr * (hi - lo)
                f2 = f(x2)
        for a, v in ((x1, f1), (x2, f2)):
            if v < v_best:
                a_best, v_best = a, v
        return a_best, v_best

    # step sizes of the vectorized line search (fw_batch=True); Frank-Wolfe often
    # needs small steps, so the grid is dense near 0
    _FW_ALPHAS = np.array(
        [1e-3, 2e-3, 5e-3, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    )

    @staticmethod
    def _son_batch(T: np.ndarray) -> np.ndarray:
        """Sum of pairwise distances for a batch of node-parameter matrices (G, n, p)."""
        sq = np.sum(T * T, axis=2)
        D2 = sq[:, :, None] + sq[:, None, :] - 2.0 * (T @ T.transpose(0, 2, 1))
        return 0.5 * np.sqrt(np.maximum(D2, 0.0)).sum(axis=(1, 2))

    def _fw_line_search(self, values, f, n_golden: int = 4):
        """Best step on the grid _FW_ALPHAS (`values` = Phi on the grid, computed in one
        batch), refined by a few golden-section steps between neighbouring grid points."""
        a_grid = self._FW_ALPHAS
        j = int(np.argmin(values))
        a_best, v_best = a_grid[j], float(values[j])
        lo = a_grid[j - 1] if j > 0 else 0.0
        hi = a_grid[min(j + 1, len(a_grid) - 1)]
        gr = 0.5 * (np.sqrt(5.0) - 1.0)
        x1, x2 = hi - gr * (hi - lo), lo + gr * (hi - lo)
        f1, f2 = f(x1), f(x2)
        for _ in range(n_golden):
            if f1 < f2:
                hi, x2, f2 = x2, x1, f1
                x1 = hi - gr * (hi - lo)
                f1 = f(x1)
            else:
                lo, x1, f1 = x1, x2, f2
                x2 = lo + gr * (hi - lo)
                f2 = f(x2)
        for a, v in ((x1, f1), (x2, f2)):
            if v < v_best:
                a_best, v_best = a, v
        return a_best, v_best

    def _omega_grad(self) -> np.ndarray:
        """Partial derivatives (105) of Phi w.r.t. omega_{v,i}, with the safeguard
        max(e_{i,v}, fw_eps) in the graph part (which is unbounded as e_{i,v} -> 0)."""
        D = np.zeros_like(self.Om)
        if self.wf > 0:
            D += self.wf * self.F
        if self.wg > 0:
            D -= self.wg * self.AE / np.maximum(self.E, self.fw_eps)
        if self.ws > 0:
            D += self.ws * (_son_subgrad(self.T) @ self.Th.T)
        return D

    def _fw_omega_step(self) -> bool:
        """Row LMO (108)-(109) for every row (the vertex with the smallest (105)),
        update (111) Omega <- (1-a) Omega + a S with one common step a chosen by line
        search on Phi.  Applied only if Phi decreases."""
        if self.Om.shape[1] == 1:
            return False
        D = self._omega_grad()
        S = np.zeros_like(self.Om)
        S[np.arange(self.n), D.argmin(1)] = 1.0
        phi0 = self.objective()
        Om0 = self.Om.copy()
        f = lambda a: self._objective_at((1 - a) * Om0 + a * S, self.Th, F=self.F)
        if self.fw_batch:
            al = self._FW_ALPHAS[:, None, None]
            Oms = (1 - al) * Om0[None] + al * S[None]  # (G, n, k)
            vals = np.zeros(len(self._FW_ALPHAS))
            if self.wg > 0:
                Es = np.sqrt(Oms)
                vals -= self.wg * np.sum(Es * (self.Abar0 @ Es), axis=(1, 2))
            if self.wf > 0:
                vals += self.wf * np.sum(Oms * self.F[None], axis=(1, 2))
            if self.ws > 0:
                vals += self.ws * self._son_batch(Oms @ self.Th)
            a, v = self._fw_line_search(vals, f)
        else:
            a, v = self._line_search_unit(f, *self.fw_ls)
        if v < phi0 - 1e-12 * max(1.0, abs(phi0)):
            self.Om = (1 - a) * Om0 + a * S
            self._refresh()
            return True
        return False

    def _fw_theta_step(self, wf=None, ws=None) -> bool:
        """Atom LMO (112) for every atom (gradient (106)), update (113)
        Theta <- (1-a) Theta + a S with one common step a chosen by line search.
        Needs `loss.lmo`, i.e. a compact Theta_f.  Applied only if Phi decreases."""
        wf = self.wf if wf is None else wf
        ws = self.ws if ws is None else ws
        if wf == 0:
            return False
        grad = wf * self.loss.grad_atoms(self.Th, self.Om)
        if ws > 0:
            grad = grad + ws * (self.Om.T @ _son_subgrad(self.T))
        S = self.loss.lmo(grad)
        Th0 = self.Th.copy()
        psi0 = self._objective_at(self.Om, Th0, wf=wf, ws=ws, graph=False)
        f = lambda a: self._objective_at(self.Om, (1 - a) * Th0 + a * S, wf=wf, ws=ws, graph=False)
        if self.fw_batch:
            vals = np.array([
                wf * float(np.sum(self.Om * self.loss.values((1 - a) * Th0 + a * S)))
                for a in self._FW_ALPHAS
            ])
            if ws > 0:
                T0, T1 = self.Om @ Th0, self.Om @ S
                al = self._FW_ALPHAS[:, None, None]
                vals = vals + ws * self._son_batch((1 - al) * T0[None] + al * T1[None])
            a, v = self._fw_line_search(vals, f)
        else:
            a, v = self._line_search_unit(f, *self.fw_ls)
        if v < psi0 - 1e-12 * max(1.0, abs(psi0)):
            self.Th = (1 - a) * Th0 + a * S
            self._refresh()
            return True
        return False

    # ------------------------------------------------------------------
    # Forward step (F.4)  (docs/solver_explained.md, Sec. 4)
    # ------------------------------------------------------------------
    def _rate_terms(self):
        """Pieces of the rate (115)-(116) that do not depend on the new atom:
        d_v(theta) = wf f_v(theta) + ws <G_v, theta> + base_v, with
        base_v = wg (Abar0 L)_vv - wf l_v - ws <G_v, theta_v>."""
        G = _son_subgrad(self.T) if self.ws > 0 else np.zeros((self.n, self.p))
        ell = np.sum(self.Om * self.F, 1)  # current lifted loss of each node
        cdiag = np.sum(self.E * self.AE, 1)  # (Abar0 L)_vv
        base = self.wg * cdiag - self.wf * ell - self.ws * np.sum(G * self.T, 1)
        return G, base

    def _d_of_theta(self, theta, G, base):
        """d_v(theta) of (116) for every node."""
        d = base.copy()
        if self.wf > 0:
            d += self.wf * self.loss.values(theta[None])[:, 0]
        if self.ws > 0:
            d += self.ws * (G @ theta)
        return d

    def _graph_B(self) -> np.ndarray:
        return self.wg * self.Abar0 if self.wg > 0 else np.zeros((self.n, self.n))

    @staticmethod
    def _rate_of_set(S, d, B) -> float:
        """Rate (115) of the full move of the node set S: sum_{v in S} d_v - 1_S^T B 1_S."""
        return float(d[S].sum() - B[np.ix_(S, S)].sum())

    def exact_rate(self, S, theta) -> float:
        """Exact one-sided derivative of Phi along (114) for the full move of the node
        set S to a new atom theta.  Equals the rate (115)-(116) plus a term for tied
        pairs (theta_v = theta_w): if exactly one node of such a pair moves, the pair
        separates and the fusion term grows like ws * t * ||theta - theta_v||, while
        (115)-(116) uses g_vw = 0.  Used by the tests, not by the oracle (which ranks
        with (115)-(116); the exact line search decides)."""
        G, base = self._rate_terms()
        val = self._rate_of_set(S, self._d_of_theta(theta, G, base), self._graph_B())
        if self.ws > 0:
            tied = squareform(pdist(self.T)) <= 1e-12
            np.fill_diagonal(tied, False)
            split = tied[np.ix_(S, ~S)]  # v in S, w not in S, theta_v = theta_w
            c = self.ws * np.linalg.norm(theta[None] - self.T[S], axis=1)
            val += float(c @ split.sum(1))
        return val

    def _sweep_set(self, score, d, B):
        """Round a nonnegative score vector to a set: among the prefixes of the nodes
        sorted by decreasing score, return the one with the smallest rate."""
        order = np.argsort(-score)
        order = order[score[order] > 1e-12]
        if order.size == 0:
            return None
        inc = d[order] - 2.0 * np.tril(B[np.ix_(order, order)], -1).sum(1)  # rate increments
        j = int(np.argmin(np.cumsum(inc)))
        S = np.zeros(self.n, dtype=bool)
        S[order[: j + 1]] = True
        return S

    def _local_search(self, S, d, B, max_flips: int | None = None):
        """Greedy single-node insertions/removals that decrease the rate (best flip first)."""
        S = S.copy()
        a = B @ S.astype(float)
        max_flips = 2 * self.n if max_flips is None else max_flips
        for _ in range(max_flips):
            marg = d - 2.0 * a  # change of the rate when v is added (minus: removed)
            gain = np.where(S, -marg, marg)
            v = int(np.argmin(gain))
            if gain[v] >= -1e-12:
                break
            S[v] = not S[v]
            a += (1.0 if S[v] else -1.0) * B[:, v]
            if not S.any():
                break
        return S

    def _theta_oracle(self, S, G):
        """Best parameter for a fixed set S (eq. (121)):
        argmin_theta sum_{v in S} [wf f_v(theta) + ws <G_v, theta>]."""
        if self.wf == 0:
            return self.Th[0].copy()
        w = self.wf * S.astype(float)
        c = self.ws * G[S].sum(0) if self.ws > 0 else np.zeros(self.p)
        return self.loss.weighted_argmin(w, c, theta0=self.Th[0])

    def _lmo(self):
        """Approximate linear minimization oracle (F.4).

        A candidate is (S, theta): every node in S moves the same fraction t of its
        mass to a new atom theta (path (114) with e = 1_S/sqrt|S|).  Candidates are
        ranked by the rate (115)-(116) of the full move,
            rate(S, theta) = sum_{v in S} d_v(theta) - wg * sum_{v != w in S} Abar0_vw.
        Search:
          1. scores s_v = d_v(theta_v^new) with every node's own best parameter (117)-(118);
          2. starting sets: the 3 nodes with the smallest s_v, and (if there is a graph)
             the best level sets of the 3 leading eigenvectors of wg*Abar0 - diag(s)
             (eq. (119)), both signs;
          3. from each start, alternate theta <- (121) and single-node flips of S.
        Returns the candidates sorted by rate, without duplicate sets."""
        G, base = self._rate_terms()
        if self.wf > 0:
            Th_v = self.loss.per_node_argmin(self.wf, self.ws * G)
            s = self.wf * self.loss.pointwise(Th_v) + base
            if self.ws > 0:
                s += self.ws * np.sum(G * Th_v, 1)
        else:
            s = base.copy()

        starts = []
        for v in np.argsort(s)[:3]:
            e = np.zeros(self.n)
            e[v] = 1.0
            starts.append(e)
        if self.wg > 0:
            M = self.wg * self.Abar0 - np.diag(s)
            _, U = np.linalg.eigh(M)
            for j in range(1, min(self.n_eig, self.n) + 1):
                for sgn in (1.0, -1.0):
                    starts.append(np.maximum(sgn * U[:, -j], 0.0))

        B = self._graph_B()
        cands = []
        for e in starts:
            S = self._sweep_set(e, s, B)
            if S is None:
                continue
            for _ in range(self.lmo_alternations):
                theta = self._theta_oracle(S, G)
                d = self._d_of_theta(theta, G, base)
                S_new = self._local_search(S, d, B)
                if not S_new.any():
                    break
                if np.array_equal(S_new, S):
                    break
                S = S_new
            theta = self._theta_oracle(S, G)
            d = self._d_of_theta(theta, G, base)
            cands.append((self._rate_of_set(S, d, B), S, theta))
        cands.sort(key=lambda c: c[0])
        out, seen = [], []
        for c in cands:  # discard duplicate supports
            if any(np.array_equal(c[1], s_) for s_ in seen):
                continue
            seen.append(c[1])
            out.append(c)
        return out

    def _path_objective(self, t, u, theta, f_new, P, sq_u):
        """Exact Phi along path (114): node v moves the fraction t*u_v of its mass to the
        new atom.  Graph term: sqrt(1-t u)^T (Abar0 o L) sqrt(1-t u) + t u^T Abar0 u."""
        a = 1.0 - t * u
        val = 0.0
        if self.wg > 0:
            sa = np.sqrt(np.maximum(a, 0.0))
            val -= self.wg * (float(sa @ P @ sa) + t * sq_u)
        if self.wf > 0:
            val += self.wf * float(np.sum(a * np.sum(self.Om * self.F, 1)) + t * (u @ f_new))
        if self.ws > 0:
            T = a[:, None] * self.T + (t * u)[:, None] * theta[None]
            val += self.ws * _son_value(T)
        return val

    def _forward(self) -> tuple[bool, float]:
        """Forward step: oracle, then exact line search on t in (0, 1] (20-point grid +
        12 golden-section steps) for the n_line_search best candidates; the best
        (candidate, t) is appended only if it lowers Phi.  Returns (added, best rate)."""
        if self.Om.shape[1] >= self.max_atoms:
            return False, 0.0
        cands = self._lmo()
        phi0 = self.objective()
        best_rate = cands[0][0] if cands else 0.0
        # The rate is a first-order model: valid only for small steps, and not the
        # exact derivative at tied parameters.  Several candidates are therefore
        # line-searched on the exact Phi, which decides.
        cands = [c for c in cands if c[0] < -1e-10 * max(1.0, abs(phi0))][: self.n_line_search]
        if not cands:
            return False, best_rate
        if self.wg > 0:
            L = self.E @ self.E.T
            P = self.Abar0 * L
        else:
            P = None
        best = (phi0, None, None, None)
        grid = np.linspace(0.0, 1.0, 21)[1:]
        for rate, S, theta in cands:
            u = S.astype(float)
            f_new = self.loss.values(theta[None])[:, 0] if self.wf > 0 else np.zeros(self.n)
            sq_u = float(u @ self.Abar0 @ u) if self.wg > 0 else 0.0
            vals = [self._path_objective(t, u, theta, f_new, P, sq_u) for t in grid]
            j = int(np.argmin(vals))
            lo, hi = grid[max(j - 1, 0)] if j > 0 else 0.0, grid[min(j + 1, len(grid) - 1)]
            gr = 0.5 * (np.sqrt(5.0) - 1.0)  # golden-section refinement
            t_best, v_best = grid[j], vals[j]
            x1, x2 = hi - gr * (hi - lo), lo + gr * (hi - lo)
            f1 = self._path_objective(x1, u, theta, f_new, P, sq_u)
            f2 = self._path_objective(x2, u, theta, f_new, P, sq_u)
            for _ in range(12):
                if f1 < f2:
                    hi, x2, f2 = x2, x1, f1
                    x1 = hi - gr * (hi - lo)
                    f1 = self._path_objective(x1, u, theta, f_new, P, sq_u)
                else:
                    lo, x1, f1 = x1, x2, f2
                    x2 = lo + gr * (hi - lo)
                    f2 = self._path_objective(x2, u, theta, f_new, P, sq_u)
            for tt, vv in ((x1, f1), (x2, f2)):
                if vv < v_best:
                    t_best, v_best = tt, vv
            if v_best < best[0]:
                best = (v_best, t_best, u, theta)
        if best[1] is None or best[0] >= phi0 - 1e-12 * max(1.0, abs(phi0)):
            return False, best_rate
        _, t, u, theta = best
        a = 1.0 - t * u
        Om = np.hstack([a[:, None] * self.Om, (t * u)[:, None]])  # path (114) at step t
        self._set_state(Om, np.vstack([self.Th, theta[None]]))
        return True, best_rate

    # ------------------------------------------------------------------
    # Backward step (F.5)  (docs/solver_explained.md, Sec. 6)
    # ------------------------------------------------------------------
    def _delete_atom(self, Om, Th, j):
        """Remove atom j: rows are renormalized proportionally; a row that sat
        entirely on atom j moves to the remaining atom closest to theta_j."""
        Om2 = np.delete(Om, j, axis=1)
        Th2 = np.delete(Th, j, axis=0)
        rs = Om2.sum(1)
        dead = rs <= 1e-12
        Om2[~dead] /= rs[~dead, None]
        if dead.any():
            near = int(np.argmin(np.linalg.norm(Th2 - Th[j], axis=1)))
            Om2[dead] = 0.0
            Om2[dead, near] = 1.0
        return Om2, Th2

    def _backward_candidates(self, n_mass: int = 4, n_dup: int = 2):
        """Atoms tried by the truncation step (cheap screen, as in Alg. 2 of CoGEnT):
        the lightest atoms and the atoms closest to another atom."""
        k = self.Om.shape[1]
        mass = self.Om.sum(0)
        cand = list(np.argsort(mass)[:n_mass])
        if k > 2:
            Dm = squareform(pdist(self.Th))
            np.fill_diagonal(Dm, np.inf)
            cand += list(np.argsort(Dm.min(1))[: 2 * n_dup])
        return list(dict.fromkeys(int(c) for c in cand))

    def _backward(self, phi_prev: float, phi_post: float) -> int:
        """Truncation rule (123): delete the best candidate atom (after one
        enhancement cycle) while Phi <= eta Phi_prev + (1 - eta) Phi_post."""
        removed = 0
        thr = self.eta * phi_prev + (1.0 - self.eta) * phi_post
        while self.Om.shape[1] > 1:
            snap = self._snapshot()
            scores = []
            cands = self._backward_candidates()
            for j in cands:
                self._set_state(*self._delete_atom(snap[0], snap[1], j))
                scores.append(self.objective())
            j = cands[int(np.argmin(scores))]
            self._set_state(*self._delete_atom(snap[0], snap[1], j))
            self._inner(self.backward_cycles)
            if self.objective() <= thr:
                removed += 1
                continue
            self._restore(snap)
            break
        return removed

    # ------------------------------------------------------------------
    # Main loop (Algorithm 1)
    # ------------------------------------------------------------------
    def solve(self, init: tuple[np.ndarray, np.ndarray] | None = None) -> dict:
        """Run Algorithm 1 and return the result dictionary (see `result`)."""
        n = self.n
        if init is not None:
            self._set_state(*init)
        else:  # one atom, every node fully on it, theta_1 = argmin sum_v f_v
            theta1 = (
                self.loss.weighted_argmin(np.ones(n), np.zeros(self.p))
                if self.wf > 0
                else np.zeros(self.p)
            )
            self._set_state(np.ones((n, 1)), theta1[None])
        phi = self.objective()
        hist = [phi]
        info = []
        for it in range(self.max_outer):
            phi_prev = phi
            added, rate = self._forward()
            self._inner(self.inner_cycles)
            phi_post = self.objective()
            removed = self._backward(phi_prev, phi_post) if self.eta > 0 else 0
            phi = self.objective()
            hist.append(phi)
            info.append(
                dict(it=it, added=added, rate=rate, removed=removed, k=self.Om.shape[1], phi=phi)
            )
            if self.verbose:
                print(
                    f"[CoGEnT] it={it:02d} k={self.Om.shape[1]:2d} phi={phi:.6f} "
                    f"rate={rate:.3e} added={added} removed={removed}"
                )
            if not added and phi_prev - phi <= 1e-9 * max(1.0, abs(phi)):
                break
        self._inner(2 * self.inner_cycles)  # final polish
        if self.wf == 0:  # graph only: compute Theta for the final Omega (mu -> 0+ limit)
            if self.enhancement == "fw":
                for _ in range(200):
                    if not self._fw_theta_step(wf=1.0, ws=self.gamma / n):
                        break
            else:
                self._theta_step(iters=50, wf=1.0, ws=self.gamma / n)
        return self.result(hist, info)

    def result(self, hist=None, info=None) -> dict:
        """Result dictionary.

        y_pred      phi(theta_v) with the node parameter theta_v = sum_i omega_vi theta_i
                    (Sec. 3.1 of the paper) -- the prediction used in the experiments;
        y_pred_atom phi of the node's largest-weight atom (equal to y_pred for one-hot rows);
        assign      index of the node's largest-weight atom;
        Omega, Theta, k (number of atoms), objective, hist (Phi per outer iteration), info."""
        assign = self.Om.argmax(1)
        dec_node = self.loss.decode(self.Om @ self.Th)
        dec_atom = self.loss.decode(self.Th)
        return dict(
            Omega=self.Om.copy(),
            Theta=self.Th.copy(),
            assign=assign,
            y_pred=dec_node,
            y_pred_atom=None if dec_atom is None else dec_atom[assign],
            k=self.Om.shape[1],
            objective=self.objective(),
            hist=np.array(hist if hist is not None else []),
            info=info or [],
        )


def cogent_solve(loss, A_graph=None, gamma=1.0, mu=np.inf, **kw) -> dict:
    """Convenience wrapper: CoGEnT(loss, A_graph, gamma, mu, **kw).solve(init)."""
    init = kw.pop("init", None)
    return CoGEnT(loss, A_graph, gamma, mu, **kw).solve(init=init)
