"""
src/theorem51_verify.py

Numerically verify the closed-form recovery conditions of
Theorems 4.1, 4.7, and 5.1 from:
  "When Graph Structure Provably Helps Classification:
   Non-Asymptotic Recovery Guarantees"

No convex program is solved.  For each (p_max, rho) on a grid we evaluate
the closed-form bounds and classify the point as one of:

  graph-only   -- Corollary 4.2   :  p <= p*_max(K)
  node-only    -- Theorem  4.7    :  rho <= gamma / K
  combined     -- Corollary 5.2   :  exists mu s.t. p < (1-Theta)/4
               -- Theorem 5.1 full:  above AND R* >= R_min(p, Theta)
  neither      -- no guarantee

This reproduces Figure 3 of the paper (K=5, gamma=2, m_i=m_v=1, d_min=2.5).

Key identities used
-------------------
zeta(r) = r^2/(2*m_v)              quadratic residual price (App. E.1)
Theta    = K*mu*zeta(r_max)         residual budget (eq. 17)
                                    => 1/(K*mu) = r_max^2/(2*m_v*Theta)
R_min(p, Theta) = (r_max^2/(2*m_v*Theta))
                  * [Theta - (1-4p)
                     + (K-2)*p/(1-p)
                     + (K-1)*p^2/((1-p)^2*(1-4p-Theta))]   (eq. 18)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ============================================================
# Paper parameters  (Figure 3)
# ============================================================
K = 5
GAMMA = 2.0  # paper's gamma  (SON coeff inside problem = gamma/n)
M_I = 1.0  # strong convexity const of class loss F_i
M_V = 1.0  # strong convexity const of node loss  f_v
D_MIN = 2.5  # min distance between biased centroids

# Derived
GAMMA_K = GAMMA / K  # node-only ceiling  rho <= gamma/K
R_STAR = M_V * D_MIN**2 / 8.0  # max R-separability (Remark 4.6)

# Graph-only threshold (Corollary 4.2)
P_STAR = ((K + 4) - np.sqrt((K + 4) ** 2 - 16)) / 8.0

print("=" * 60)
print("Theorem 5.1 verification  —  Figure 3 of the paper")
print("=" * 60)
print(f"  K={K},  gamma={GAMMA},  m_i={M_I},  m_v={M_V},  d_min={D_MIN}")
print(f"  Graph-only threshold  p*_max  = {P_STAR:.4f}  (Corollary 4.2)")
print(f"  Node-only ceiling     gamma/K = {GAMMA_K:.4f}  (Theorem 4.7)")
print(f"  R-separability bound  R*      = {R_STAR:.5f}  (Remark 4.6)")
print()


# ============================================================
# Grid  (p_max in [0,0.25], rho in [0,1.5])
# ============================================================
N_P = 400
N_RHO = 400
p_vals = np.linspace(1e-6, 0.25, N_P)
rho_vals = np.linspace(0.0, 1.5, N_RHO)
PP, RR = np.meshgrid(p_vals, rho_vals)  # (N_RHO, N_P)

r_max_map = np.maximum(0.0, RR - GAMMA_K)  # residual variability

# ------------------------------------------------------------------
# Graph-only  (Corollary 4.2)
# ------------------------------------------------------------------
graph_only = PP <= P_STAR

# ------------------------------------------------------------------
# Node-only  (Theorem 4.7 lower bound)
# ------------------------------------------------------------------
node_only = RR <= GAMMA_K

standalone = graph_only | node_only

# ------------------------------------------------------------------
# Combined  —  Corollary 5.2  (p-condition only)
#
# For fixed (p, rho):
#   r_max = max(0, rho - gamma/K)
#   Theta  = K * mu * r_max^2 / (2*m_i)
#   Corollary 5.2: p < (1-Theta)/4
#
# Minimize Theta by taking mu = mu_min (= 1e-2):
#   Theta_min = K * mu_min * r_max^2 / (2*m_i)
# For r_max = 0: Theta = 0 for any mu => p < 1/4 always.
# ------------------------------------------------------------------
MU_MIN = 1e-2
MU_MAX = 1e3
N_MU = 300
MU_GRID = np.logspace(np.log10(MU_MIN), np.log10(MU_MAX), N_MU)

# p < 1/4 is the theoretical maximum (Theta -> 0)
Theta_min_map = K * MU_MIN * r_max_map**2 / (2.0 * M_I)
comb_52 = (PP < (1.0 - Theta_min_map) / 4.0) & (PP < 0.25)
comb_52 |= (r_max_map == 0.0) & (PP < 0.25)

gain_52 = comb_52 & ~standalone

# ------------------------------------------------------------------
# Combined  —  Theorem 5.1 full  (p-condition AND R* >= R_min)
#
# Vectorize over (mu, rho, p):  shape (N_MU, N_RHO, N_P)
# ------------------------------------------------------------------
print(
    "Computing Theorem 5.1 full check (vectorized over mu grid)... ", end="", flush=True
)

MU3 = MU_GRID[:, None, None]  # (N_MU, 1,     1    )
PP3 = PP[None, :, :]  # (1,    N_RHO, N_P  )
RM3 = r_max_map[None, :, :]  # (1,    N_RHO, N_P  )

Theta3 = K * MU3 * RM3**2 / (2.0 * M_I)  # (N_MU, N_RHO, N_P)
slack3 = (1.0 - 4.0 * PP3) - Theta3  # needs to be > 0

# Valid entry: p-condition holds AND interval (0, 1-4p) non-empty
valid3 = (
    (PP3 > 0)
    & (PP3 < 0.25)
    & (Theta3 > 0)
    & (slack3 > 0)
    & (PP3 < (1.0 - Theta3) / 4.0)
)

# R_min bracket (eq. 18) — only for r_max > 0
# Paper formula:  Theta - (1-4p) + (K-2)*p/(1-p) + (K-1)*p^2/((1-p)^2*(1-4p-Theta))
rm_pos = valid3 & (RM3 > 0)
bracket3 = np.where(
    rm_pos,
    (
        Theta3
        - (1.0 - 4.0 * PP3)
        + (K - 2) * PP3 / np.where(PP3 < 1, 1.0 - PP3, 1.0)
        + (K - 1)
        * PP3**2
        / (np.where(PP3 < 1, 1.0 - PP3, 1.0) ** 2 * np.where(slack3 > 0, slack3, 1.0))
    ),
    0.0,
)
prefac3 = np.where(
    rm_pos,
    RM3**2 / (2.0 * M_V * np.where(Theta3 > 0, Theta3, 1.0)),
    0.0,
)
Rmin3 = np.where(rm_pos, prefac3 * bracket3, np.inf)

# r_max = 0: mu -> inf gives Theta -> 0, R_min -> 0  (automatically satisfied)
Rmin3 = np.where(valid3 & (RM3 == 0.0), 0.0, Rmin3)

# Full condition: valid p AND R*>=R_min
thm51_3 = valid3 & (R_STAR >= Rmin3)  # (N_MU, N_RHO, N_P)

# Union over mu axis
comb_51 = thm51_3.any(axis=0)  # (N_RHO, N_P)
gain_51 = comb_51 & ~standalone

print("done.")


# ============================================================
# Statistics
# ============================================================
def frac(m):
    return 100.0 * m.sum() / PP.size


sa = frac(standalone)
print()
print("-" * 60)
print("Recovery region areas  (% of the full (p_max, rho) grid)")
print("-" * 60)
print(f"  Graph-only         (Corollary 4.2) : {frac(graph_only):6.2f}%")
print(f"  Node-only          (Theorem  4.7)  : {frac(node_only):6.2f}%")
print(f"  Standalone union                   : {sa:6.2f}%")
print(f"  Combined           (Corollary 5.2) : {frac(comb_52):6.2f}%")
print(f"  Combined           (Theorem  5.1)  : {frac(comb_51):6.2f}%")
g52 = 100.0 * gain_52.sum() / standalone.sum()
g51 = 100.0 * gain_51.sum() / standalone.sum()
print(f"  Gain vs standalone (Corollary 5.2) : {g52:6.2f}%  of standalone")
print(f"  Gain vs standalone (Theorem  5.1)  : {g51:6.2f}%  of standalone")
print()
print("  Paper reports ~22% gain  (green region / standalone union area).")
print("-" * 60)


# ============================================================
# Limiting checks  (Theorem 5.1 remarks)
# ============================================================
print()
print("Limiting checks:")

# mu -> 0  => Theta -> 0  => combined threshold -> 1/4 (Corollary 5.2)
# This means the combined can tolerate p < 1/4 for rho <= gamma/K
print(f"  mu -> 0 limit : Theta -> 0 => combined p-threshold = 1/4 = 0.25")
print(f"                  Graph-only threshold p*_max = {P_STAR:.4f}")
print(f"                  Ratio (combined/graph-only) = {0.25/P_STAR:.2f}x")

# mu -> inf => Theta = K*mu*r_max^2/(2m) -> inf (if r_max>0), but must stay < 1-4p
# For r_max=0: Theta = 0, p < 1/4 (node-only already certifies rho<=gamma/K)
print(f"  mu -> inf limit (rho <= gamma/K): Theta -> 0, p < 1/4 — same as mu->0")

# Graph-only regime: p*_max (Corollary 4.2)
print(f"  Graph-only recovers for p <= {P_STAR:.4f}   (any rho)")
print(f"  Node-only recovers  for rho <= {GAMMA_K:.4f}  (any p)")
print(f"  Combined (Cor 5.2) recovers for p < 0.25   AND rho (any)")
print(f"  => combined extends p-tolerance by factor {0.25/P_STAR:.2f}x")


# ============================================================
# Figure (Figure 3 analogue)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)

labels = [
    "Corollary 5.2  ($p < (1-\\Theta)/4$, $\\Theta$ via $\\mu_{\\min}$)",
    "Theorem 5.1 full  (Corollary 5.2  $+$  $R^* \\geq R_{\\min}$)",
]
gmaps = [gain_52, gain_51]
cmaps = [comb_52, comb_51]
gains_pct = [g52, g51]

ext = [p_vals[0], p_vals[-1], rho_vals[0], rho_vals[-1]]

for ax, lbl, gmap, cmap, gp in zip(axes, labels, gmaps, cmaps, gains_pct):
    ax.set_facecolor("white")

    # Light blue: standalone union
    blue = np.zeros((*PP.shape, 4))
    blue[standalone] = [0.45, 0.65, 1.0, 0.70]
    ax.imshow(blue, origin="lower", extent=ext, aspect="auto")

    # Green: combined gain only
    green = np.zeros((*PP.shape, 4))
    green[gmap] = [0.15, 0.78, 0.15, 0.85]
    ax.imshow(green, origin="lower", extent=ext, aspect="auto")

    # Threshold lines
    ax.axvline(
        P_STAR,
        color="red",
        ls="--",
        lw=2,
        label=f"$p^*_{{\\max}}$={P_STAR:.3f}  (Cor. 4.2, graph-only)",
    )
    ax.axhline(
        GAMMA_K,
        color="blue",
        ls="--",
        lw=2,
        label=f"$\\gamma/K$={GAMMA_K:.2f}  (Thm. 4.7, node-only)",
    )
    ax.axvline(
        0.25,
        color="green",
        ls=":",
        lw=1.5,
        label="$p=1/4$  (combined limit $\\Theta\\to 0$)",
    )

    ax.set_xlabel("$p_{\\mathrm{max}}$ (graph misconnection rate)", fontsize=12)
    ax.set_ylabel("$\\rho$ (within-class variability)", fontsize=12)
    ax.set_title(lbl, fontsize=10, pad=7)
    ax.set_xlim(p_vals[0], p_vals[-1])
    ax.set_ylim(rho_vals[0], rho_vals[-1])

    # Gain annotation
    ax.text(
        0.97,
        0.97,
        f"Gain: {gp:.1f}%  of standalone\n(paper ≈ 22%)",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", fc="white", alpha=0.85, ec="gray"),
    )

    patches = [
        mpatches.Patch(
            color=[0.45, 0.65, 1.0], label="Standalone  (graph-only $\\cup$ node-only)"
        ),
        mpatches.Patch(
            color=[0.15, 0.78, 0.15], label="Combined gain  (Theorem 5.1 only)"
        ),
        mpatches.Patch(color="white", ec="gray", label="No guarantee"),
    ]
    ax.legend(handles=patches, fontsize=8, loc="upper left", framealpha=0.9)

fig.suptitle(
    f"Recovery regions in the $(p_{{\\max}},\\rho)$ plane  "
    f"[$K={K}$, $\\gamma={GAMMA}$, $m={M_I}$, $d_{{\\min}}={D_MIN}$,  "
    f"$\\mu \\in [{MU_MIN:.0e},{MU_MAX:.0e}]$]\n"
    f"$p^*_{{\\max}}={P_STAR:.3f}$,  $\\gamma/K={GAMMA_K:.2f}$,  "
    f"$R^*={R_STAR:.4f}$",
    fontsize=11,
)
plt.tight_layout()

os.makedirs(os.path.join(os.path.dirname(__file__), "..", "results"), exist_ok=True)
out = os.path.join(
    os.path.dirname(__file__), "..", "results", "theorem51_verification.png"
)
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nFigure saved → {os.path.abspath(out)}")
plt.show()
