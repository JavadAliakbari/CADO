# Changes with respect to the original code

This file lists **every change** made to the original solver and experiment code, with the
reason for each change. Equation numbers refer to the paper (Appendix F unless stated
otherwise). The math behind every step is derived in
[`docs/solver_explained.md`](docs/solver_explained.md); this file says *what changed and why*.

**Original code** means the files of the project before this work:

* the solver: `cogent.py`, `update_W.py`, `update_A.py`, `update_Pi.py`;
* the experiments: `src/cogent_empirical_validation.py`, `src/sweep_p_rho.py`.

They are not part of this repository.

---

## Summary

| # | part | original code | new code |
| --- | --- | --- | --- |
| 1 | node loss | squared loss hard-wired; label term handled separately | any convex $f_v$ through the `NodeLoss` interface; the label-aware loss (22) is one instance |
| 2 | initialization | zero atoms; first atom = a data point | $k=1$, $\Omega=\mathbf 1$, $\theta_1=\arg\min\sum_v f_v$ (Alg. 1, line 1) |
| 3 | forward step | atom at the data point with the largest gradient; every node moves the fixed fraction $1/(k+1)$, so $\Phi$ can increase | oracle over (node set $S$, parameter $\theta$) ranked by the rate (115)–(116); exact line search along (114) on the 6 best candidates; atom added only if $\Phi$ decreases |
| 4 | enhancement step | projected gradient on $W$ that ignores the label term; separate updates of $A$ and $\Pi$ | **two options**, both on the full $\Phi$: `"bcd"` (default; exact row moves + majorize–minimize on $\Theta$) and `"fw"` (Frank–Wolfe exactly as in F.3) |
| 5 | backward step | truncation evaluated without the label term; $\eta=0.9$ hard-coded | truncation rule (123) on the full $\Phi$, $\eta=0.5$, cheap candidate screen |
| 6 | number of atoms | at most 20 outer iterations, so at most 20 atoms | up to $2n$ outer iterations, no cap on atoms |
| 7 | prediction | `W.argmax`, then `pred % K` | $\hat y_v=\phi(\theta_v)$ with $\theta_v=\sum_i\omega_{v,i}\theta_i$ (Sec. 3.1) |
| 8 | regimes | node only: CoGEnT; graph only: spectral clustering | all three regimes are the same program (14) and the same solver; only $\mu$ changes |
| 9 | experiment | unlabeled clustering; "combined" = best $\mu$ in hindsight | 5 labels per class, accuracy on the unlabeled nodes; $\mu$ chosen by cross-validation on the training labels |
| 10 | graph generator | only $p_{\max}\in\{0.04,0.12,0.20,\dots\}$ reachable; the same banded graph for every seed | every $p_{\max}=m/25$ reachable; random node order per seed |
| 11 | checks | none | tests of the rate formula, of the generic loss path, of both enhancement options, and a comparison with the ground-truth objective |

The sections below give the details.

---

## 1. The objective and the node loss

**Before.** The squared loss $\tfrac12\|x_v-a\|^2$ was hard-wired in `update_W.py` and
`update_A.py`. The label term of (22) was added as a separate block (`beta`, `Pi`), with its own
update. The fusion (sum-of-norms) term acted only on the feature block $a$, not on the full
parameter $\theta=(a,\pi)$.

**Now.** The solver minimizes the reduced objective (104), divided by $n\mu$:

$$
\Phi(\Omega,\Theta)= -w_g\sum_{u\ne v}\bar A^0_{uv}L_{uv}
+ w_f\sum_{v,i}\omega_{v,i}f_v(\theta_i)
+ w_s\sum_{u<v}\|\theta_u-\theta_v\|_2 ,
$$

with $L=\sqrt\Omega\sqrt\Omega^{\top}$, $\theta_v=\sum_i\omega_{v,i}\theta_i$, and
$(w_g,w_f,w_s)=(1/(n\mu),1,\gamma/n)$ (combined), $(0,1,\gamma/n)$ (node only, $\mu=\infty$),
$(1,0,0)$ (graph only, $\mu=0$).

* The node losses enter **only** through the `NodeLoss` interface (`cogent/losses.py`). A new
  loss needs `values`, `pointwise`, `grad_atoms` and `smoothness` (plus `project` if
  $\Theta_f$ is constrained). Everything else has a generic fallback, which closed forms can
  override.
* Two losses are provided: `SquaredLoss` and `LabelAwareLoss` (22),
  $f_v(a,\pi)=\tfrac12\|x_v-a\|^2-\nu\,\mathbb 1\{v\text{ observed}\}\,\pi_{\tilde y_v}$ with
  $\pi\in\Delta^K$.
* The fusion term acts on the full $\theta=(a,\pi)$, so labels propagate through it.
* `tests/test_general_loss.py` runs the solver with a loss that has no closed forms (the generic
  path reproduces `SquaredLoss` exactly) and with a constrained Huber loss.

**Why.** App. F describes a solver for general convex $f_v$, and the paper's experiments use the
label-aware loss (22). The original code solved a different problem (squared loss, label term
outside the fused parameter).

## 2. Initialization (Alg. 1, lines 1–2)

**Before.** The experiment started from zero atoms (`init_with_one_atom=False`); the first atom
was the data point with the largest gradient norm.

**Now.** One atom, every node fully on it ($\Omega=\mathbf 1$), and
$\theta_1=\arg\min_\theta\sum_v f_v(\theta)$, as in Alg. 1.

## 3. Forward step (F.4)

**Before.** The new atom was placed at $x_{v^*}$ with $v^*=\arg\max_v\|q_v\|$, and *every* node
moved the fixed fraction $\rho=1/(k+1)$ of its mass to it (the direction $e=\mathbf 1_n/\sqrt n$).
Along that direction the objective is affine in the step, so no line search was possible, and
the forward step could **increase** the objective. App. F says so in the paragraph
"Implementation used in Sec. 6".

**Now** (`CoGEnT._forward`, `_lmo`):

1. **What a candidate is.** A candidate is a pair $(S,\vartheta)$: every node in the set $S$
   moves the same fraction $t$ of its mass to a new atom with parameter $\vartheta$. This is path
   (114) with $e_v^2\propto\mathbf 1\{v\in S\}$.
2. **Ranking.** Candidates are ranked by the paper's rate (115)–(116) of the *full* move,
   $R(S,\vartheta)=\sum_{v\in S}d_v(\vartheta)-w_g\sum_{v\ne w\in S}\bar A^0_{vw}$.
3. **Search.**
   * Starting sets: the best level sets of the 3 leading eigenvectors of
     $w_g\bar A^0-\mathrm{diag}(s)$ (eq. (119)), both signs, and the 3 single nodes with the
     smallest scores $s_v$ (117)–(118).
   * From each start, alternate the $\theta$-oracle (121) and single-node insertions/removals
     of $S$ that decrease $R$.
4. **Decision.** The 6 candidates with the most negative $R$ are line-searched on the **exact**
   $\Phi$ along (114): 20 grid points on $t\in(0,1]$, then 12 golden-section steps. The best
   (candidate, $t$) is added only if it lowers $\Phi$.

**Why the set $S$ (instead of the unit vector $e$ of (119)).** Along (114) a node can give at
most all of its mass, so the admissible step is $\rho\le 1/\max_v e_v^2$, which depends on $e$.
Comparing the rate per unit $\rho$ over unit-norm $e$ compares candidates at different step lengths
and favours atoms concentrated on one or two nodes. CoGEnT compares atoms at the boundary of the
feasible set, i.e. at the full step. With $u_v=e_v^2/\max_w e_w^2\in[0,1]$ the full-step rate is
$-\sum\bar A_{vw}\sqrt{u_vu_w}+\sum_v u_vd_v$. The code searches over $u=\mathbf 1_S$. Without a
graph this is exact: every node that the new atom fits better moves.

**Why 6 candidates and an exact line search.** The rate (115)–(116) is correct only when no two
node parameters are tied. At ties (every class at a hard assignment is a tied group) the exact
derivative has an extra nonnegative term, and (115) can even have the wrong sign
(`CoGEnT.exact_rate`, checked in `tests/test_forward_rate.py`). The rate is in any case only a
first-order model. With only the steepest 3 candidates line-searched, the solver stalled on noisy
graphs: $\Phi=797$ where the true partition has $495$. Line-searching 6 fixed this. Ranking by
the exact (tie-aware) rate instead was tried and was worse (`docs/solver_explained.md`, Sec. 4.4).

## 4. Enhancement step (F.3): two options

**Before.** Projected gradient on $W$ (`pg_max_iter=10`, step 0.05) that ignored the label
term, unconstrained gradient descent on the atoms $A$, and a separate update of $\Pi$.

**Now.** Both options work on the full $\Phi$, with the active set fixed, and never increase
$\Phi$. Choose one with `CoGEnT(..., enhancement="bcd" | "fw")`.

### 4a. `enhancement="bcd"` (default; used for the main figure)

* **Rows of $\Omega$** (`_omega_sweep`; numba kernel `cogent/_numba_kernels.py`). Block-coordinate
  descent over the rows, in random order. For row $v$ the candidate rows are:
  * the full moves to the 6 atoms with the smallest partial derivative (105), plus the atoms the
    node already uses;
  * the Frank–Wolfe points $(1-\alpha)\omega_v+\alpha\delta_{i^*}$, $\alpha\in\{0.25,0.5\}$;
  * the pairwise moves of half or all of the mass of the worst active atom to $i^*$.

  Only row and column $v$ of $L$ and $\theta_v$ change, so the change of $\Phi$ is computed
  **exactly** for every candidate. The best is applied if it lowers $\Phi$.
* **Atoms $\Theta$** (`_theta_step`). $\Phi$ is convex in $\Theta$ for fixed $\Omega$.
  Majorize–minimize on the fusion term,
  $\|z\|\le\|z\|^2/(2\|z_0\|)+\|z_0\|/2$. For (22) the surrogate is a $k\times k$ linear system
  for the feature block plus a simplex-constrained QP (FISTA) for the label block. A step is
  accepted only if the true objective decreases.

### 4b. `enhancement="fw"` (Frank–Wolfe exactly as written in F.3)

* **Rows of $\Omega$** (`_fw_omega_step`). Row LMO (108)–(109): the vertex with the smallest
  partial derivative (105). Update (111) with one common step for all rows.
* **Atoms $\Theta$** (`_fw_theta_step`). Atom LMO (112) with the gradient (106), update (113)
  with one common step.
* **Step sizes.** Line search on $\Phi$. The default (`fw_batch=True`) evaluates the grid
  $\{0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.15,0.2,0.3,\dots,1\}$ in one vectorized batch, then
  runs 4 golden-section steps. A sequential grid + golden search is available with
  `fw_batch=False`. Each update is applied only if $\Phi$ decreases.
* **Two details F.3 does not state but the code needs.**
  * The atom LMO (112) over $a\in\mathbb R^d$ is unbounded. The code uses
    $\Theta_f=\{\|a\|\le R\}\times\Delta^K$ with $R=\max_v\|x_v\|$ (`LabelAwareLoss(a_radius=R)`).
    This does not change the optimal value of (14), because projecting the feature blocks onto
    the convex hull of the features does not increase any term. The experiments use the same
    ball for **both** options, so they solve the same problem.
  * The graph part of (105), $-(\bar A e_i)_v/e_{i,v}$, is unbounded as $e_{i,v}\to0$. The code
    uses $\max(e_{i,v},\varepsilon)$ with $\varepsilon=10^{-8}$ (`fw_eps`).

### 4c. How the two options compare (Fig. 3 setup)

* Both give a non-increasing $\Phi$ and recover easy instances (`tests/test_enhancements.py`).
  Both reproduce the synergy of the combined regime (Sec. 10 below).
* With 3 cycles per outer iteration, `"fw"` reaches clearly worse node-only solutions when the
  classes fragment into many atoms (accuracy 0.40 vs 0.78 at $\rho=4$). It is not uniformly
  worse on the objective: on the 45 ground-truth instances it ends above `"bcd"` in 30 cases, but
  below it on average at $\mu=0.03$ (`docs/solver_explained.md`, Sec. 8.3). It is also about 2.2×
  slower per trial. With 10 cycles node only is close to `"bcd"`, but the run is 3.3× slower.
* The $\varepsilon$ safeguard makes the graph part of (105) of order $w_g/\varepsilon$ on every
  empty entry, so the graph term dominates the row LMO even at large $\mu$. At $\mu=100$ FW
  ended 4–5 above the node-only objective; $\varepsilon=10^{-2}$ removes the gap but is an
  arbitrary constant. The exact-change row moves of `"bcd"` do not have this sensitivity.
* **Recommendation:** `"bcd"` for the paper's figure; `"fw"` is kept as the literal
  implementation of F.3.

## 5. Backward step (F.5)

**Before.** The truncation test evaluated the objective **without** the label term, with
`eta=0.9` hard-coded.

**Now** (`_backward`). Truncation rule (123) on the full $\Phi$: delete atoms while
$\Phi\le\eta\Phi_{\rm prev}+(1-\eta)\Phi_{\rm post}$, with $\eta=0.5$.

* A cheap screen picks the deletion candidate, as in Alg. 2 of CoGEnT. It evaluates $\Phi$
  without re-optimization for the 4 lightest atoms and the 4 atoms closest to another atom.
* Only the best candidate is re-optimized (one enhancement cycle) before the test.
* When an atom is removed, its rows are renormalized. A row that sat entirely on the removed atom
  moves to the nearest remaining atom.

## 6. Termination, atoms and output

* **Atom budget.** Before: at most 20 outer iterations (`max_iters=20`), hence at most 20 atoms.
  That capped the node-only regime, whose optimum needs many atoms when $\rho>\gamma/K$. Now up
  to $2n$ outer iterations and no cap (`max_atoms=n`).
* **Stopping.** The loop stops when no atom was added and $\Phi$ decreased by less than $10^{-9}$
  (relative), followed by 6 polishing enhancement cycles.
* **Graph only.** With $\mu=0$ the node terms vanish. $\Theta$ is computed at the end for the
  final $\Omega$ with weights $(w_f,w_s)=(1,\gamma/n)$, i.e. the $\mu\to0^+$ limit.
* **Prediction.** Before: `W.argmax` (the node's heaviest atom), then `pred % K`, which merged
  extra atoms arbitrarily. Now $\hat y_v=\phi(\theta_v)$ with $\theta_v=\sum_i\omega_{v,i}\theta_i$,
  as defined in Sec. 3.1 of the paper; for (22), $\arg\max_c\pi_{v,c}$, with ties broken at random
  (fixed seed). The heaviest-atom prediction differs only for soft rows of $\Omega$, but those
  are common with `"fw"`: there it dropped node-only accuracy at $\rho=1$ from 1.00 to 0.10.
* **Accuracy.** `clustering_accuracy` uses an optimal one-to-one (Hungarian) matching; unmatched
  clusters count as errors. The experiments report plain test accuracy of $\hat y_v$, because
  the labels fix the class names.

## 7. Monotonicity

$\Phi$ never increases across outer iterations. The forward step is accepted only if it lowers
$\Phi$; every enhancement update is accepted only if it lowers $\Phi$; the truncation rule keeps
$\Phi\le\Phi_{\rm prev}$. The original code did not have this property, because of the fixed
forward step. Global optimality is **not** guaranteed: the oracle is approximate and the reduced
problem is not convex in $\Omega$.

## 8. Regimes

**Before.** Node only was the CoGEnT code with $\mu=\infty$. Graph only was spectral clustering of
$A$ (`src/sweep_p_rho.py`). Combined was CoGEnT with the graph term.

**Now.** All three are the same convex program (14), solved by the same algorithm:

| regime | $\mu$ | $(w_g,w_f,w_s)$ |
| --- | --- | --- |
| node only | $\infty$ | $(0,1,\gamma/n)$ |
| graph only | $0$ | $(1,0,0)$, $\Theta$ decoded at the end |
| combined | $0<\mu<\infty$ | $(1/(n\mu),1,\gamma/n)$ |

## 9. Experiment (Fig. 3, center and right)

Code: `experiments/fig3_sweeps.py` and `experiments/synthetic.py`.

* **Setup.** $K=5$, 25 nodes per class, centroids on a circle of radius 4, $d=2$.
  * 5 observed labels per class; accuracy on the 100 unlabeled nodes.
  * Loss (22) with $\nu=2$; $\gamma=2$.
  * Center panel: $\rho=4$, $p_{\max}\in\{0.04,\dots,0.44\}$. Right panel: $p_{\max}=0.36$,
    $\rho\in\{1,\dots,6\}$.
  * 20 trials (BCD) and 10 trials (FW).

  The original experiments were unlabeled clustering (no observed labels): $K=3$ with 50 nodes
  per class in `src/cogent_empirical_validation.py`, and $K=5$ with 10 nodes per class in
  `src/sweep_p_rho.py`.
* **Features** (`make_clusters_fixed_rho`, unchanged). Points on the circle of radius $\rho/2$
  around each centroid, including one antipodal pair, so the class diameter is exactly $\rho$.
* **Graph** (`construct_controlled_graph`, two changes). It is regular: circulant inside
  classes, shifted bipartite between classes, so $\delta_{ij}=0$ and $p_{\max}$ is exact.
  1. *Rounding.* With 25 nodes per class the original rounded the intra-class misconnection up,
     so only $p_{\max}\in\{0.04,0.12,0.20,0.28,0.36\}$ was reachable (a target of 0.15 became
     0.20). It is now rounded down, so $p_{\max}=m/25$ exactly for every $m$.
  2. *Random node order.* The node order is permuted at random per seed, inside every class and
     for every pair of classes, before the blocks are laid out. Before, the graph was the same
     banded graph for every seed, and graph only was either perfect or ≈0.15 (below chance) for
     all seeds alike. Now graph only degrades gradually: 1.00 up to $p_{\max}=0.32$, then 0.47,
     0.31, 0.23. `RANDOM_GRAPH = False` restores the old graph.
* **Choice of $\mu$.** Before: a single $\mu$, or the best $\mu$ in hindsight (using the
  evaluation labels) over a small grid ($\{1.2,2.2\}$ in `src/cogent_empirical_validation.py`).
  Now:
  * $\mu$ is chosen per instance from $\{0.01,0.03,0.1,1,10,100\}$ by stratified 3-fold
    cross-validation on the **training labels only**;
  * ties go to the $\mu$ whose neighbour-averaged CV score is largest (the interior of the
    plateau of best values);
  * the best-$\mu$-in-hindsight curve is still drawn, dashed, as a reference envelope.

  The tie-break rule was changed once, after seeing results ("ties → largest $\mu$" picked the
  edge of the plateau). This should be reported as such.
* **Random streams.** Split: `seed`. Features: (`seed`, $\rho$). Graph: (`seed`, $p$). So along
  the $\rho$ axis the graph and the split are fixed, and along the $p$ axis the features are
  fixed.
* **Speed.** One BLAS thread per worker process, numba kernel for the BCD row sweep, vectorized
  line search for FW. The main BCD run takes 93 min on 11 processes.
* **Minor fix.** `--mu-select` is now passed to the worker processes. Before, workers always ran
  the CV, which only cost time; the default results are unaffected.

## 10. Results of the final runs (`results/`)

Combined (CV) is at or above both baselines at all 22 points of the BCD run.

| | node only | graph only | combined (CV) |
| --- | --- | --- | --- |
| center, $p_{\max}=0.36$ / 0.40 / 0.44 ($\rho=4$) | 0.777 | 0.466 / 0.309 / 0.225 | **1.000 / 0.982 / 0.831** |
| right, $\rho=1$ … 6 ($p_{\max}=0.36$) | 1.000 → 0.579 | ≈0.47 | **1.000 → 0.968** |
| FW run, center, $p_{\max}=0.36$ / 0.40 / 0.44 | 0.397 | 0.535 / 0.293 / 0.202 | 0.983 / 0.964 / 0.847 |
| FW run, right, $\rho=1$ … 6 | 1.000 → 0.243 | ≈0.54 | 0.992 → 0.944 |

With FW, combined is slightly below FW's own node only at $\rho\le1.5$ (0.99/0.98 vs
1.00/0.99), because of the weaker FW solutions (Sec. 4c). Full tables:
`docs/solver_explained.md`, Sec. 10.

## 11. Verification

| check | file | result |
| --- | --- | --- |
| rate (115)–(116) vs finite differences (no ties) | `tests/test_forward_rate.py` | rel. error ≤ 1.5e-6 |
| exact rate with the tie term vs finite differences (hard assignments) | `tests/test_forward_rate.py` | rel. error ≤ 2.4e-6 (without the term: up to 4.2, wrong sign) |
| generic loss path = closed form; constrained Huber loss runs | `tests/test_general_loss.py` | passes |
| both enhancement options: $\Phi$ non-increasing; easy instance recovered | `tests/test_enhancements.py` | passes |
| $\Phi(\text{solver})\le\Phi(\text{ground-truth partition})$ on 45 instances | `tests/check_vs_ground_truth.py` | never above; largest relative excess $-1.2\cdot10^{-10}$ (BCD), $-2.2\cdot10^{-3}$ (FW) |
| this repository reproduces the saved runs | re-run of 9 saved jobs (node, graph, combined + CV; BCD and FW) | identical accuracies, $\mu$ choices and CV scores |

## 12. Suggested changes to the paper

The review of App. F (what is correct, what should change) and ready-to-paste LaTeX are in
[`docs/paper_notes/`](docs/paper_notes/). In short:

* (115)–(116) is exact only without ties; state the tie term as a caveat, and let the line
  search on $\Phi$ decide;
* normalize the oracle by the full step (the set $S$) instead of unit-norm $e$;
* "$\Phi_k$ is convex with the active set fixed" is false in $\Omega$; it is convex in $\Theta$;
* the claim that the method "drives the active set to exactly $K$ atoms" is not proven; state it
  as an empirical observation;
* if F.3 keeps Frank–Wolfe, add the compact $\Theta_f$ and the $\varepsilon$ safeguard;
* remove the leftovers of the track-changes merge listed in the review.
