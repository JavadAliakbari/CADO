# Review of Appendix F (atomic conditional-gradient solver)

Equation numbers refer to the NeurIPS camera-ready draft (App. F, pp. 38–42).
Code references are to this repository (`cogent/solver.py`, `tests/`); the full derivation of
every step is in [`../solver_explained.md`](../solver_explained.md).

## 1. What is correct

* **Reduced parameterization (102)–(104).** Correct. Row-stochastic `Omega`,
  `L(Omega) = sum_i sqrt(Omega_:i) sqrt(Omega_:i)^T` and `theta_v = sum_i omega_vi theta_i`
  give `L_vv = 1`, `0 <= L_uv <= 1`, and a lifted node loss that is linear in `Omega`.
* **Gradients (105)–(107).** Correct (chain rule `d omega = 2 e de` for the graph term).
* **Rate of a candidate atom (115)–(116).** Correct *when no two node parameters coincide*.
  It is the exact one-sided derivative of (104) along path (114) in that case; see 2(a0) for the
  tied case.
  `tests/test_forward_rate.py` compares it with finite differences of the
  objective for the squared loss and the label-aware loss (22), at
  `mu in {0.3, 2, inf}`; the largest relative error is about 1e-6.
* **Rate zero for active atoms at a stationary point.** Correct (with `d_v(theta_j) = (Abar e_j)_v / e_jv`
  on the support of `e_j`).
* **Lower bound from per-node candidates (117)–(118).** Correct.
* **Weighted average (120).** It is the exact minimizer over `theta` for the squared loss only, as the
  text already says. For (22) the `pi`-part of the exact minimizer (121) is a simplex vertex.
* **The "implementation used in Sec. 6" paragraph.** It correctly describes the old code: with
  `e = 1_n/sqrt(n)` the objective is affine in the step, so a line search would return 0. That is
  why the old code's forward step increased the objective.

## 2. What should be changed

### (a0) The rate at tied node parameters (most important)

(115)–(116) linearize the fusion term with `g_vu = 0` when `theta_v = theta_u`. The text says the
true one-sided derivative "can exceed (115) by a nonnegative term, which the line search accounts
for". The line search does protect monotonicity, but it does not fix the *oracle*, which still
ranks candidates by the wrong rate.

Ties are not a corner case. At hard assignments (perfect recovery, and most iterates near it),
every class is a tied group. Moving part of a group to a new atom separates tied pairs, which
costs `gamma mu |e_v^2 - e_u^2| ||theta - theta_v||` per pair to first order. The exact rate is

    rate_exact(e, theta) = (115) + gamma*mu * sum_{v<u: theta_v = theta_u} |e_v^2 - e_u^2| * ||theta - theta_v||.

In `tests/test_forward_rate.py` (hard assignments), (115) alone has relative error up to 4.2 and
gets the sign wrong (-7.76 vs +2.45). With the tie term the error is 2e-6.

Without the term, the oracle returned "descent" atoms that the line search rejected, and the
solver stopped. On a noisy graph (`p_max = 0.36`, `mu = 0.03`) it stalled at 2 atoms with
objective 797, while the true partition has 495.

For full moves (`u = 1_S`) the term is a quadratic set function, `c_g s_g (n_g - s_g)` per tied
group, so it fits into the same eigenvector, level-set and local-search machinery as the graph
term.

**What this means for the algorithm.** At tied parameters the rate (115) is only a heuristic
ranking. Even the exact rate is only a first-order model, valid for small steps. The code
therefore ranks candidates with (115) but line-searches the 6 best on the exact `Phi`, instead of
only the steepest one. It adds an atom only if `Phi` decreases.

Ablation (details in `docs/solver_explained.md`, Sec. 4.4):

* line-searching only the top 3 candidates stalled twice on noisy graphs;
* line-searching 6 fixed this;
* ranking by the exact (tie-aware) rate instead was worse.

Suggested wording for F.4:

* "the oracle proposes candidates; the line search on `Phi` decides";
* the stopping rule becomes "no candidate decreases `Phi`" rather than "the rate is nonnegative";
* state the tie remark as a caveat on (115).

### (a) Normalization of the oracle (119)

The step along (114) is limited by `rho <= 1/max_v e_v^2`, and that limit depends on `e`.
Minimizing the rate per unit `rho` over unit-norm `e` (eq. (119)) therefore compares candidates at
different step lengths. It systematically prefers atoms concentrated on one or two nodes, because the
per-unit rate is a weighted *average* of `d_v` over the support.

The Frank–Wolfe / CoGEnT analogue of `<grad f(x_t), tau a - x_t>` is the first-order change at the
**maximal** feasible step, `rate(e) / max_v e_v^2`. With `u_v := e_v^2 / max_w e_w^2 in [0,1]`
(the fraction of node `v`'s mass moved to the new atom), the oracle is

    min_{u in [0,1]^n, theta}  - sum_{v,w} Abar_vw sqrt(u_v u_w) + sum_v u_v d_v(theta).        (119')

Its spectral relaxation still uses the leading eigenvector of `Abar - diag(s)`, so the eigenvector
computation is unchanged. Only the rounding and the comparison change:

* rescale the (nonnegative part of the) eigenvector to `max_v e_v = 1`;
* take the level set `{v : e_v >= tau}` with the most negative value of (119');
* alternate with the `theta`-step (121) using weights `u_v`.

In the node-only case (`Abar` absent, no tied parameters) the exact solution for fixed `theta` is `u_v = 1{d_v(theta) < 0}`:
every node that the new atom fits better moves. That is the CoGEnT oracle of (4).

The inexact-oracle condition (11) of CoGEnT is stated for this full-step quantity.

### (b) The inner problem is not convex (F.3, first sentence)

"With the active set fixed, `Phi_k` is convex" is false.

* The graph term is `-sum_i sum_{u != w} Abar_uw sqrt(omega_ui omega_wi)`.
  * For a non-edge (`Abar_uw = -1`) the summand `+sqrt(omega_ui omega_wi)` is **concave**.
  * Example with two nodes and one non-edge: `2[sqrt(ab) + sqrt((1-a)(1-b))]`.
* The fusion term `|| sum_i (omega_ui - omega_wi) theta_i ||` is not jointly convex in `(Omega, Theta)`.

What is true:

* `Phi_k` is convex in `Theta` for fixed `Omega`.
* The node-loss and fusion parts are convex in `Omega` for fixed `Theta`.
* The convexity of (14)/(46) is in the coefficients `lambda` over the *fixed, infinite* atom set,
  not in the reduced variables.

Suggested wording: "With the active set fixed we decrease `Phi_k` by block-coordinate descent ...
(`Phi_k` is convex in `{theta_i}` for fixed `Omega`, but not in `Omega` because of the graph term)."

### (b2) The Frank–Wolfe enhancement step as written in F.3

The code implements F.3 as written (`enhancement="fw"`), next to the default block-coordinate
step (`enhancement="bcd"`). Running it needs two details that the text does not state:

* **Compact `Theta_f`.** The atom LMO (112) over `a in R^d` is unbounded (a linear function has no
  minimizer). The code uses `||a|| <= R` with `R = max_v ||x_v||`. This does not change the optimal
  value of (14), because projecting the feature blocks onto the convex hull of the features does
  not increase any term. The same ball is used for both enhancement options, so they solve the
  same problem.
* **Safeguard in (105).** The graph part `-(Abar e_i)_v / e_{i,v}` is unbounded as `e_{i,v} -> 0`;
  the code uses `max(e_{i,v}, eps)` with `eps = 1e-8`.

Observed behaviour (Fig. 3 setup):

* The safeguard makes the graph part of order `w_g / eps` on every empty entry, so it dominates the
  row LMO however large `mu` is. At `mu = 100` FW ended 4–5 above the node-only objective (banded-graph run, 3 seeds);
  with `eps = 1e-2` the gap disappears, but `eps` is then an arbitrary tuning constant.
* With 3 enhancement cycles per outer iteration, FW's node-only solutions are clearly worse than
  BCD's when the classes fragment into many atoms (accuracy 0.40 vs 0.78 at `rho = 4`). With 10
  cycles they are close, but the run is 3.3x slower than BCD. Even with 3 cycles FW is 2.2x slower
  per trial.
* Combined (CV) with FW is 0.94–1.00 along both sweeps, close to BCD.

Suggestion: describe the block-coordinate step in F.3 (text (A) in
`appendix_F_suggested_text.tex`), or keep FW and add the two details above (text (B)).

### (c) Claims in the introduction of App. F and at the end of F.5

* "converges monotonically in the objective": fine. Forward with line search, the inner step and
  truncation rule (123) are all non-increasing. The old code did **not** have this property
  (fixed forward step).
* "produces a K-atom representation that, under the conditions of Thm. 5.1, coincides with the
  perfect-recovery solution" and "In the regimes covered by Thm. 5.1, the truncation step drives
  the active set to exactly K atoms": not proven. The reduced problem is nonconvex and the oracle is
  approximate.
  * Safer: "the objective is non-increasing, and in our experiments the algorithm returns the K-atom
    perfect-recovery solution whenever the conditions of Thm. 5.1 hold".
  * The new code checks the empirical claim directly: on every instance tested, the objective the
    solver reaches is at or below the objective of the ground-truth partition.

### (d) Nonnegativity of `e`

Restricting atoms to `e >= 0` replaces `{L psd, 0 <= L <= 1}` by the completely-positive cone. This
is a restriction (inner approximation) of (46), still convex. The perfect-recovery solution (15)
has `e >= 0`, so the certificate of Thm. 5.1 carries over. One sentence saying so would help.

### (e) Leftovers of the track-changes merge

The rendered PDF contains old and new text side by side. Keep only one of each:

| Location | Text to resolve |
| --- | --- |
| F.4, after (116) | the dangling clause "where q(t)v is the residual gradient (107) at the current iterate" |
| Alg. 1, line 1 | `e_1 <- 1_n/sqrt(n)` and `e_1 <- 1_n` |
| Alg. 1, line 2 | `Phi_prev <- +inf` and `Phi_1(...)` |
| Alg. 1, lines 6–7 | "append the atom by (122); k <- k+1" appears twice |
| F.4, "Forward update" | "rescaling existing weights by 1 - rho_t to preserve column-stochasticity" (old) next to "along the path (114)" (new) |
| F.5 | "Atom j is pruned whenever" next to "The atom j* with the smallest ... is pruned if" |

### (f) "Implementation used in Sec. 6"

The new solver (`cogent/solver.py`) implements the forward step as described: oracle, `theta`-alternation
(121), line search along (114). This paragraph can therefore be replaced by a short description of
the oracle rounding in (a).

Ready-to-paste LaTeX for F.3 (first paragraph, and the implementation paragraph for either
enhancement option), F.4 ("Solving the oracle", "Forward update"), F.5 (candidate screen) and
Algorithm 1 is in [`appendix_F_suggested_text.tex`](appendix_F_suggested_text.tex).

### (g) Existence of (117)

The per-node problem `argmin_theta n mu f_v(theta) + gamma mu <G_v, theta>` needs `f_v` to be
coercive or `Theta_f` to be compact. This holds for (22): quadratic in `a`, simplex for `pi`.

## 3. What the old code did differently from App. F

| App. F | old `cogent.py` |
| --- | --- |
| initial atom `theta_1 = argmin sum_v f_v`, `Omega = 1` | starts from zero atoms; first atom = farthest data point |
| oracle (117)–(121), line search along (114) | atom at `x_{v*}`, `v* = argmax ||q_v||`; fixed step `1/(k+1)` for every node (objective goes up) |
| inner step on the full objective | W-update ignores the label term (22); `Pi` updated separately; SON only on the feature block, not on `theta = (a, pi)` |
| truncation (123) on the full objective | truncation evaluates the objective **without** the label term; `eta = 0.9` hard-coded |
| general `f_v` | squared loss hard-wired in `update_W.py` / `update_A.py` |
| clustering accuracy | `pred % K` in `cluster_accuracy` merges extra atoms arbitrarily |
| (n/a) | at most `max_iters = 20` atoms, which caps the node-only regime (its optimum needs many atoms when `rho > gamma/K`) |

## 4. Notes for Sec. 6 (experiments)

* **Generator granularity.** With 25 nodes per class, `construct_controlled_graph` in
  `src/cogent_empirical_validation.py` (original experiment code, not in this repository) only reaches `p_max in {0.04, 0.12, 0.20, 0.28, 0.36}`.
  An odd class size forces an even intra-class degree, and the intra block is rounded *up*.
  A target of 0.15 is therefore realized as `p_max = 0.20`.
  * `experiments/synthetic.py` uses the same construction with the intra block rounded *down*, so
    `p_max = m/25` exactly.
  * Figures should use the realized `p_max` on the x-axis.
* **Graph-only baseline.** This is now the same program and solver with `mu -> 0`, instead of
  spectral clustering.
  * With the original (deterministic, banded) regular graph it was perfect up to `p_max = 0.20`
    and failed completely (accuracy about 0.15) from `p_max = 0.24`, identically for every seed.
  * The final runs permute the node order at random per seed before laying out the regular
    blocks (`p_max` stays exact). Graph only is then perfect up to `p_max = 0.32` and degrades
    gradually: 0.47, 0.31, 0.23 at `p_max = 0.36, 0.40, 0.44`.
  * At small `p_max` graph-only is perfect, so the combined curve can only tie it. The rho-sweep
    therefore uses `p_max = 0.36`, where graph only is partially informative (about 0.47).
* **Choice of mu.** Thm. 5.1 is an envelope over `mu`, and the best `mu` depends on `(p_max, rho)`.
  The script selects `mu` per instance from `{0.01, 0.03, 0.1, 1, 10, 100}` by 3-fold
  cross-validation on the observed training labels only. It also reports the best `mu` in the
  grid (envelope) as a dashed line.
* **Setup used.** K = 5, 25 nodes per class, ring radius 4, 5 observed labels per class,
  gamma = 2, nu = 2 in (22), test accuracy, 20 trials. The node-only optimum fragments into many
  atoms when `rho > gamma/K`, so no cap is placed on the number of atoms.

## 5. Final results

The final Fig. 3 results (BCD: 20 seeds; FW: 10 seeds) and their interpretation are in
`docs/solver_explained.md`, Sec. 10, and the figures are in `results/enhancement_bcd/` and
`results/enhancement_fw/`.
