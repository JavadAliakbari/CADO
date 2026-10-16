"""
Optional numba kernel for the default enhancement step ("bcd"): one pass of
block-coordinate descent over the rows of Omega.

It is a loop-level translation of `CoGEnT._omega_sweep` (cogent/solver.py): the
same candidate rows (screened vertices, two Frank-Wolfe points, two pairwise
points) and the same exact change of Phi for each candidate.  It updates
Om, E, T and AE in place and returns the total decrease of Phi.  The solver
imports it only if numba is installed; otherwise the numpy version is used.

Arguments: Om (n,k) Omega, E = sqrt(Om), Th (k,p) Theta, T = Om @ Th,
AE = Abar0 @ E, F (n,k) f_v(theta_i), Ab = Abar0, (wg, wf, ws) weights of Phi,
order = node order of the sweep, n_vc = number of vertex candidates.
"""

import numpy as np
from numba import njit


@njit(cache=True)
def omega_sweep(Om, E, Th, T, AE, F, Ab, wg, wf, ws, order, n_vc):
    n, k = Om.shape
    p = Th.shape[1]
    total = 0.0
    dist_v = np.empty(n)
    G_v = np.empty(p)
    D = np.empty(k)
    Wc = np.empty((k + 4, k))
    Tc = np.empty(p)
    sel = np.empty(k, dtype=np.bool_)
    for idx in range(n):
        v = order[idx]
        sum_dv = 0.0
        if ws > 0.0:
            for c in range(p):
                G_v[c] = 0.0
            for u in range(n):
                if u == v:
                    dist_v[u] = 0.0
                    continue
                s = 0.0
                for c in range(p):
                    d = T[v, c] - T[u, c]
                    s += d * d
                s = np.sqrt(s)
                dist_v[u] = s
                sum_dv += s
                if s > 1e-12:
                    for c in range(p):
                        G_v[c] += (T[v, c] - T[u, c]) / s
        # partial derivatives (105): only used to pick the FW / pairwise directions
        for i in range(k):
            val = wf * F[v, i]
            if wg > 0.0:
                val -= wg * AE[v, i] / max(E[v, i], 1e-8)
            if ws > 0.0:
                s = 0.0
                for c in range(p):
                    s += Th[i, c] * G_v[c]
                val += ws * s
            D[i] = val
        i_to = 0
        for i in range(1, k):
            if D[i] < D[i_to]:
                i_to = i
        j_away = -1
        for i in range(k):
            if Om[v, i] > 1e-12 and (j_away < 0 or D[i] > D[j_away]):
                j_away = i
        # candidate rows: vertices (screened by D), FW steps, pairwise steps
        if k > n_vc:
            for i in range(k):
                sel[i] = Om[v, i] > 1e-12
            srt = np.argsort(D)
            for t in range(n_vc):
                sel[srt[t]] = True
        else:
            for i in range(k):
                sel[i] = True
        nc = 0
        for i in range(k):
            if sel[i]:
                for j in range(k):
                    Wc[nc, j] = 0.0
                Wc[nc, i] = 1.0
                nc += 1
        for a in (0.25, 0.5):
            for j in range(k):
                Wc[nc, j] = (1.0 - a) * Om[v, j]
            Wc[nc, i_to] += a
            nc += 1
        if j_away != i_to:
            for a in (0.5, 1.0):
                for j in range(k):
                    Wc[nc, j] = Om[v, j]
                amt = a * Om[v, j_away]
                Wc[nc, j_away] -= amt
                Wc[nc, i_to] += amt
                nc += 1
        best_d = -1e-12
        best_c = -1
        for c in range(nc):
            s = 0.0
            for j in range(k):
                if Wc[c, j] < 0.0:
                    Wc[c, j] = 0.0
                s += Wc[c, j]
            for j in range(k):
                Wc[c, j] /= s
            dl = 0.0
            for j in range(k):
                if wg > 0.0:
                    dl -= 2.0 * wg * (np.sqrt(Wc[c, j]) - E[v, j]) * AE[v, j]
                if wf > 0.0:
                    dl += wf * (Wc[c, j] - Om[v, j]) * F[v, j]
            if ws > 0.0:
                for cc in range(p):
                    s = 0.0
                    for j in range(k):
                        s += Wc[c, j] * Th[j, cc]
                    Tc[cc] = s
                dsum = 0.0
                for u in range(n):
                    if u == v:
                        continue
                    s = 0.0
                    for cc in range(p):
                        d = Tc[cc] - T[u, cc]
                        s += d * d
                    dsum += np.sqrt(s)
                dl += ws * (dsum - sum_dv)
            if dl < best_d:
                best_d = dl
                best_c = c
        if best_c >= 0:
            total += best_d
            for j in range(k):
                ec = np.sqrt(Wc[best_c, j])
                de = ec - E[v, j]
                if wg > 0.0 and de != 0.0:
                    for u in range(n):
                        AE[u, j] += Ab[u, v] * de
                Om[v, j] = Wc[best_c, j]
                E[v, j] = ec
            for cc in range(p):
                s = 0.0
                for j in range(k):
                    s += Om[v, j] * Th[j, cc]
                T[v, cc] = s
    return total
