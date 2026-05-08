"""
Verify Asimov procedure on the Jeans analysis.
Two questions:
  (1) Does M(r_1/2) recovery look right?  (Already +0.04 dex bias in summary;
      look at chain shape directly to confirm symmetry & σ.)
  (2) Where does the +0.15-0.23 dex Asimov J-factor bias come from?
      Tests:
       (a) MAP vs median vs truth on log10 J(α_c) — isolates post-chain
           transformation skew (Jensen-on-log10 J).
       (b) Marginal projection: distribution of (2 log ρ_s + 3 log r_s)
           on the chain, compared to the J(α_c) bias.  This is the
           "small-x analytic" piece.
       (c) D bias scaling: D ~ ρ_s · r_s · h(r_t/r_s).  In log-linear
           projection the D-bias coefficient is roughly half the J one.
       (d) Bias decomposition by 1D projection: hold log ρ_s at truth,
           push log r_s through; vice versa.  Sums vs full bias diagnoses
           whether non-additivity is small.
       (e) Prior-edge diagnostic: chain mass at the U(-2, 1) bounds.
"""

import json
from pathlib import Path
import numpy as np

from dwarfjeans.jeans import solver as jeans
from dwarfjeans.jd import factors as jd

OUT_DIR = Path(__file__).resolve().parents[2] / "results" / "tests" / "ufd_population"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
chain = np.load(OUT_DIR / "compact_ufd_asimov.npz", allow_pickle=True)
samples = chain["samples_eq"]        # (N, 4): V, log10_rs, log10_rhos, beta_tilde
param_names = list(chain["param_names"])
truth = dict(zip(chain["truth_keys"], chain["truth_vals"]))

N = samples.shape[0]
log_rs = samples[:, 1]
log_rhos = samples[:, 2]
beta_tilde = samples[:, 3]

# Truth
log_rs_t = float(truth["log10_rs"])
log_rhos_t = float(truth["log10_rhos"])
r_s_t = float(truth["r_s"])
rho_s_t = float(truth["rho_s"])
r_p = float(truth["r_p"])
r_half_3d_t = float(truth["r_half_3d"])
log_M3_t = float(truth["log10_M_half_3d"])
log_M2_t = float(truth["log10_M_half_2d"])

# JD geometry — Stage 2 MC convention (synthetic data carries no host
# distance or 3D position to drive a Springel+08 r_t). The historical
# ufd_asimov_jd.json (produced by the now-removed run_jd_summary.py)
# carried the same defaults; we set them inline so analyze_asimov.py is
# self-contained against the current pipeline.
d_kpc = 30.0
r_t = 1.0
alpha_c = jd.alpha_c_radians(r_half_3d_t, d_kpc)

print("=" * 70)
print("CHAIN OVERVIEW")
print("=" * 70)
print(f"  N samples = {N}")
print(f"  truth: log10_rs = {log_rs_t:+.4f}, log10_rhos = {log_rhos_t:+.4f}")
print(f"         r_s = {r_s_t} kpc, rho_s = {rho_s_t:.3e} Msun/kpc^3")
print(f"         r_half_3d = {r_half_3d_t:.4f} kpc, alpha_c = {np.degrees(alpha_c):.4f} deg")
print(f"         r_t = {r_t} kpc, d = {d_kpc} kpc")


# ---------------------------------------------------------------------------
# (1) M(r_1/2, 3D) chain shape — direct diagnostic
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("(1) M(r_1/2) RECOVERY")
print("=" * 70)
# Compute M(r_1/2, 3D) at fixed r_half_3d = truth value, evaluated at each
# chain (r_s, rho_s).  This is what `summarize_jd` and the Asimov summary use.
r_s_chain = 10.0 ** log_rs
rho_s_chain = 10.0 ** log_rhos
M3_chain = jeans.nfw_M(r_half_3d_t, r_s_chain, rho_s_chain)
log_M3_chain = np.log10(M3_chain)

# MAP via 1D KDE peak
from scipy.stats import gaussian_kde
kde_M3 = gaussian_kde(log_M3_chain)
grid_M3 = np.linspace(log_M3_chain.min(), log_M3_chain.max(), 2001)
MAP_log_M3 = float(grid_M3[np.argmax(kde_M3(grid_M3))])

print(f"  truth log M(r_1/2,3D) = {log_M3_t:.4f}")
print(f"  chain median          = {np.median(log_M3_chain):.4f}  "
      f"(offset {np.median(log_M3_chain) - log_M3_t:+.4f} dex)")
print(f"  chain mean            = {log_M3_chain.mean():.4f}  "
      f"(offset {log_M3_chain.mean() - log_M3_t:+.4f} dex)")
print(f"  chain MAP (1D KDE)    = {MAP_log_M3:.4f}  "
      f"(offset {MAP_log_M3 - log_M3_t:+.4f} dex)")
print(f"  chain std             = {log_M3_chain.std():.4f}  (Asimov σ ~ 0.14)")
# Skewness
m = log_M3_chain.mean()
s = log_M3_chain.std()
skew = ((log_M3_chain - m) ** 3).mean() / s ** 3
kurt = ((log_M3_chain - m) ** 4).mean() / s ** 4 - 3.0
print(f"  skewness              = {skew:+.3f}  (0 = symmetric)")
print(f"  excess kurtosis       = {kurt:+.3f}  (0 = Gaussian)")

# Quantile asymmetry: (q84 − med) − (med − q16)
q16, q50, q84 = np.percentile(log_M3_chain, [16, 50, 84])
print(f"  q16 / q50 / q84       = {q16:.4f} / {q50:.4f} / {q84:.4f}")
print(f"  asymmetry (q84-q50) - (q50-q16) = {(q84-q50)-(q50-q16):+.4f} dex")


# ---------------------------------------------------------------------------
# Push chain through J/D at α_c (and the four reporting angles for
# cross-check vs ufd_asimov_jd.json) - use full chain.
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("(2) PUSH FULL CHAIN THROUGH J/D")
print("=" * 70)
# Use larger n_R, n_u than the sped-up settings to reduce numerical noise:
# this is a one-off chain push, cheap enough.
n_R, n_u = 96, 256
DEG = np.pi / 180.0
angles = {
    "0p1deg": 0.1 * DEG,
    "0p2deg": 0.2 * DEG,
    "0p5deg": 0.5 * DEG,
    "alphac": alpha_c,
}

# truth J/D
J_truth = {}
D_truth = {}
for tag, th in angles.items():
    J, D = jd.J_D_factors(th, d_kpc, r_s_t, rho_s_t, r_t, n_R=n_R, n_u=n_u)
    J_truth[tag] = J
    D_truth[tag] = D

# Chain push at each angle
log_J_chain = {tag: np.empty(N) for tag in angles}
log_D_chain = {tag: np.empty(N) for tag in angles}
for i in range(N):
    for tag, th in angles.items():
        J, D = jd.J_D_factors(th, d_kpc, r_s_chain[i], rho_s_chain[i], r_t,
                               n_R=n_R, n_u=n_u)
        log_J_chain[tag][i] = np.log10(J) if J > 0 else -np.inf
        log_D_chain[tag][i] = np.log10(D) if D > 0 else -np.inf

print(f"  (chain push complete, N={N}, n_R={n_R}, n_u={n_u})")
print()
print(f"  {'angle':>10}  {'truth':>9}  {'med-truth':>9}  {'mean-truth':>10}  "
      f"{'MAP-truth':>9}  {'sigma':>6}  J-channel")
print(f"  {'-'*70}")
for tag in angles:
    lj = log_J_chain[tag]
    kde = gaussian_kde(lj)
    grid = np.linspace(lj.min(), lj.max(), 2001)
    MAP = float(grid[np.argmax(kde(grid))])
    print(f"  {tag:>10}  {np.log10(J_truth[tag]):>9.4f}  "
          f"{np.median(lj) - np.log10(J_truth[tag]):>+9.4f}  "
          f"{lj.mean() - np.log10(J_truth[tag]):>+10.4f}  "
          f"{MAP - np.log10(J_truth[tag]):>+9.4f}  "
          f"{lj.std():>6.3f}")
print()
print(f"  {'angle':>10}  {'truth':>9}  {'med-truth':>9}  {'mean-truth':>10}  "
      f"{'MAP-truth':>9}  {'sigma':>6}  D-channel")
print(f"  {'-'*70}")
for tag in angles:
    ld = log_D_chain[tag]
    kde = gaussian_kde(ld)
    grid = np.linspace(ld.min(), ld.max(), 2001)
    MAP = float(grid[np.argmax(kde(grid))])
    print(f"  {tag:>10}  {np.log10(D_truth[tag]):>9.4f}  "
          f"{np.median(ld) - np.log10(D_truth[tag]):>+9.4f}  "
          f"{ld.mean() - np.log10(D_truth[tag]):>+10.4f}  "
          f"{MAP - np.log10(D_truth[tag]):>+9.4f}  "
          f"{ld.std():>6.3f}")


# ---------------------------------------------------------------------------
# (2a) Small-x analytic test: distribution of 2 log10 rho_s + 3 log10 r_s
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("(2a) SMALL-x ANALYTIC TEST: log10(rho_s^2 r_s^3)")
print("=" * 70)
# In the small-x limit (R_max << r_s), J ~ const * rho_s^2 * r_s^3.
# So log10 J = 2 log10 rho_s + 3 log10 r_s + offset.
# Truth-relative: Delta log10 J ~ 2 d(log rho_s) + 3 d(log r_s).
small_x_chain = 2.0 * log_rhos + 3.0 * log_rs
small_x_truth = 2.0 * log_rhos_t + 3.0 * log_rs_t
delta = small_x_chain - small_x_truth
print(f"  truth 2 log rhos + 3 log rs = {small_x_truth:.4f}")
print(f"  chain median offset         = {np.median(delta):+.4f} dex")
print(f"  chain mean offset           = {delta.mean():+.4f} dex")
print(f"  chain std                   = {delta.std():.4f} dex")
print()
print("  Compare to J biases above. If small-x dominates, these should match")
print("  the smallest-aperture J bias (0.1deg) most closely (R_max smallest).")
print(f"  J(0.1deg) median offset = {np.median(log_J_chain['0p1deg']) - np.log10(J_truth['0p1deg']):+.4f} dex")


# ---------------------------------------------------------------------------
# (2b) Linear-projection test: Delta log J vs (2 d log rhos + 3 d log rs) point-by-point
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("(2b) PER-SAMPLE: log J vs small-x prediction")
print("=" * 70)
print("  For each chain point, compare actual delta_log10_J to the small-x")
print("  small-x prediction 2*d_log_rhos + 3*d_log_rs. Slope and corr indicate")
print("  how 'small-x linear' the relation actually is over the chain.")
print()
print(f"  {'angle':>10}  {'slope':>7}  {'corr':>6}  {'r2':>6}  {'<DJ-pred>':>10}")
for tag in angles:
    dJ = log_J_chain[tag] - np.log10(J_truth[tag])
    pred = 2.0 * (log_rhos - log_rhos_t) + 3.0 * (log_rs - log_rs_t)
    # OLS slope (forced through origin? no — free intercept)
    slope, intercept = np.polyfit(pred, dJ, 1)
    corr = float(np.corrcoef(pred, dJ)[0, 1])
    r2 = corr ** 2
    resid = dJ - pred
    print(f"  {tag:>10}  {slope:>7.4f}  {corr:>6.4f}  {r2:>6.4f}  "
          f"{resid.mean():>+10.4f}")


# ---------------------------------------------------------------------------
# (2c) D linear-projection: D ~ rho_s * r_s in small-x → 1*d_log_rhos + 1*d_log_rs
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("(2c) D-CHANNEL LINEAR-PROJECTION TEST")
print("=" * 70)
print("  Small-x: D ~ rho_s * r_s, so delta log D ~ 1*d_log_rhos + 1*d_log_rs.")
print()
print(f"  {'angle':>10}  {'slope':>7}  {'corr':>6}  {'r2':>6}  {'<DD-pred>':>10}")
for tag in angles:
    dD = log_D_chain[tag] - np.log10(D_truth[tag])
    pred = 1.0 * (log_rhos - log_rhos_t) + 1.0 * (log_rs - log_rs_t)
    slope, intercept = np.polyfit(pred, dD, 1)
    corr = float(np.corrcoef(pred, dD)[0, 1])
    r2 = corr ** 2
    resid = dD - pred
    print(f"  {tag:>10}  {slope:>7.4f}  {corr:>6.4f}  {r2:>6.4f}  "
          f"{resid.mean():>+10.4f}")

# Also compare median offsets: J median bias vs D median bias coefficient ratio
print()
print("  J-vs-D bias scaling (small-x predicts roughly 2x for log_rhos channel,")
print("  3x for log_rs channel, so for symmetric-channel chain, J/D ~ 2-3x):")
for tag in angles:
    bj = np.median(log_J_chain[tag]) - np.log10(J_truth[tag])
    bd = np.median(log_D_chain[tag]) - np.log10(D_truth[tag])
    if abs(bd) > 1e-4:
        print(f"  {tag:>10}: J_bias/D_bias = {bj/bd:>5.2f}   (J={bj:+.3f}, D={bd:+.3f})")
    else:
        print(f"  {tag:>10}: D_bias too small to ratio  (J={bj:+.3f}, D={bd:+.3f})")


# ---------------------------------------------------------------------------
# (2d) Marginal projection: hold one channel at truth, push the other
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("(2d) MARGINAL-PROJECTION DECOMPOSITION at α_c")
print("=" * 70)

# r_s varies, rho_s held at truth
log_J_rs_only = np.empty(N)
for i in range(N):
    J, _ = jd.J_D_factors(alpha_c, d_kpc, r_s_chain[i], rho_s_t, r_t, n_R=n_R, n_u=n_u)
    log_J_rs_only[i] = np.log10(J)
bias_rs_only = np.median(log_J_rs_only) - np.log10(J_truth["alphac"])

# rho_s varies, r_s held at truth
log_J_rhos_only = np.empty(N)
for i in range(N):
    J, _ = jd.J_D_factors(alpha_c, d_kpc, r_s_t, rho_s_chain[i], r_t, n_R=n_R, n_u=n_u)
    log_J_rhos_only[i] = np.log10(J)
bias_rhos_only = np.median(log_J_rhos_only) - np.log10(J_truth["alphac"])

bias_full = np.median(log_J_chain["alphac"]) - np.log10(J_truth["alphac"])

print(f"  full bias (joint)       = {bias_full:+.4f} dex")
print(f"  log r_s-only marginal   = {bias_rs_only:+.4f} dex")
print(f"  log rho_s-only marginal = {bias_rhos_only:+.4f} dex")
print(f"  sum (would equal full if log-linear projection) = "
      f"{bias_rs_only + bias_rhos_only:+.4f} dex")
print(f"  non-additive remainder  = {bias_full - bias_rs_only - bias_rhos_only:+.4f} dex")
print()
print("  log r_s 1D-marginal stats:")
print(f"    chain log_rs:  median offset = {np.median(log_rs) - log_rs_t:+.4f}, "
      f"std = {log_rs.std():.4f}, skew = "
      f"{((log_rs - log_rs.mean())**3).mean() / log_rs.std()**3:+.3f}")
print("  log rho_s 1D-marginal stats:")
print(f"    chain log_rhos: median offset = {np.median(log_rhos) - log_rhos_t:+.4f}, "
      f"std = {log_rhos.std():.4f}, skew = "
      f"{((log_rhos - log_rhos.mean())**3).mean() / log_rhos.std()**3:+.3f}")


# ---------------------------------------------------------------------------
# (2e) Prior-edge diagnostic
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("(2e) PRIOR-EDGE DIAGNOSTIC")
print("=" * 70)
# Priors (from jeans_inference.py — confirm by looking at chain support)
LO_RS, HI_RS = -2.0, 1.0
print(f"  log r_s prior U({LO_RS}, {HI_RS}); chain range "
      f"[{log_rs.min():.3f}, {log_rs.max():.3f}]")
for thr in [0.05, 0.10, 0.20]:
    f_lo = float(np.mean(log_rs < LO_RS + thr))
    f_hi = float(np.mean(log_rs > HI_RS - thr))
    print(f"    fraction within {thr:.2f} of edge: lo={f_lo:.3f}  hi={f_hi:.3f}")

# Same for rho_s
print(f"  log rho_s chain range [{log_rhos.min():.3f}, {log_rhos.max():.3f}]")
print("  (prior on log rho_s presumed wider; check edge-touching anyway)")


# ---------------------------------------------------------------------------
# (2f) MAP-of-2D vs truth at α_c — does the joint posterior peak sit at truth?
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("(2f) JOINT 2D KDE: where does the (log_rs, log_rhos) posterior peak?")
print("=" * 70)
kde2 = gaussian_kde(np.vstack([log_rs, log_rhos]))
# Search on a grid
gx = np.linspace(log_rs.min(), log_rs.max(), 121)
gy = np.linspace(log_rhos.min(), log_rhos.max(), 121)
XX, YY = np.meshgrid(gx, gy)
ZZ = kde2(np.vstack([XX.ravel(), YY.ravel()])).reshape(XX.shape)
imax = np.unravel_index(np.argmax(ZZ), ZZ.shape)
MAP_rs, MAP_rhos = float(XX[imax]), float(YY[imax])
print(f"  truth      log_rs = {log_rs_t:+.4f}, log_rhos = {log_rhos_t:+.4f}")
print(f"  2D KDE MAP log_rs = {MAP_rs:+.4f}, log_rhos = {MAP_rhos:+.4f}")
print(f"  marginal medians:   {np.median(log_rs):+.4f}, "
      f"{np.median(log_rhos):+.4f}")
# J at the 2D MAP vs truth
J_at_MAP, _ = jd.J_D_factors(alpha_c, d_kpc, 10.0**MAP_rs, 10.0**MAP_rhos, r_t,
                               n_R=n_R, n_u=n_u)
print(f"  log J at 2D MAP - truth = {np.log10(J_at_MAP) - np.log10(J_truth['alphac']):+.4f} dex")
print("  (this isolates where the joint posterior peak is in J-space, "
      "vs. the chain-medianed log J)")


# ---------------------------------------------------------------------------
# Done — also save chain-derived values for any plotting / further inspection
# ---------------------------------------------------------------------------
np.savez(OUT_DIR / "asimov_chain_diagnostics.npz",
         log_rs=log_rs, log_rhos=log_rhos,
         log_M3_chain=log_M3_chain,
         log_J_alphac=log_J_chain["alphac"],
         log_J_0p1=log_J_chain["0p1deg"],
         log_J_0p2=log_J_chain["0p2deg"],
         log_J_0p5=log_J_chain["0p5deg"],
         log_D_alphac=log_D_chain["alphac"],
         small_x_proxy=small_x_chain,
         truth_log_J_alphac=np.log10(J_truth["alphac"]),
         truth_log_M3=log_M3_t,
         truth_log_rs=log_rs_t,
         truth_log_rhos=log_rhos_t)
print()
print(f"saved chain-derived diagnostics to {OUT_DIR / 'asimov_chain_diagnostics.npz'}")


# ---------------------------------------------------------------------------
# Optional: triptych figure (M(r_1/2) marginal, joint halo posterior, log J marginal)
# ---------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib not available, skipping plot")
else:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # (a) M(r_1/2, 3D) marginal
    ax = axes[0]
    ax.hist(log_M3_chain, bins=60, color="0.7", edgecolor="0.4", density=True)
    ax.axvline(log_M3_t, color="C3", lw=2, label=f"truth = {log_M3_t:.3f}")
    ax.axvline(np.median(log_M3_chain), color="C0", lw=2, ls="--",
               label=f"median = {np.median(log_M3_chain):.3f}")
    ax.set_xlabel(r"$\log_{10} M(r_{1/2}, 3D)\ [M_\odot]$")
    ax.set_ylabel("density")
    ax.set_title("(a) M(r_{1/2}) recovery — clean")
    ax.legend(fontsize=8, frameon=False)

    # (b) Joint (log_rs, log_rhos) posterior
    ax = axes[1]
    kde2 = gaussian_kde(np.vstack([log_rs, log_rhos]))
    gx = np.linspace(-2.05, 1.05, 121)
    gy = np.linspace(log_rhos.min() - 0.2, log_rhos.max() + 0.2, 121)
    XX, YY = np.meshgrid(gx, gy)
    ZZ = kde2(np.vstack([XX.ravel(), YY.ravel()])).reshape(XX.shape)
    ax.contourf(XX, YY, ZZ, levels=14, cmap="Greys")
    ax.scatter(log_rs, log_rhos, s=2, alpha=0.15, c="C0")
    ax.axvline(-2.0, color="C1", ls=":", alpha=0.7, label="log r_s prior bounds")
    ax.axvline(1.0, color="C1", ls=":", alpha=0.7)
    ax.scatter([log_rs_t], [log_rhos_t], color="C3", s=120, marker="*",
               label="truth", zorder=5, edgecolor="k", linewidths=0.6)
    imax = np.unravel_index(np.argmax(ZZ), ZZ.shape)
    ax.scatter([XX[imax]], [YY[imax]], color="C2", s=80, marker="P",
               label="2D KDE peak", zorder=5, edgecolor="k", linewidths=0.6)
    ax.set_xlabel(r"$\log_{10} r_s$")
    ax.set_ylabel(r"$\log_{10} \rho_s$")
    ax.set_title("(b) Joint posterior in halo plane")
    ax.legend(fontsize=8, frameon=False, loc="lower left")

    # (c) log J(alpha_c) marginal
    ax = axes[2]
    lj = log_J_chain["alphac"]
    ax.hist(lj, bins=60, color="0.7", edgecolor="0.4", density=True)
    ax.axvline(np.log10(J_truth["alphac"]), color="C3", lw=2,
               label=f"truth = {np.log10(J_truth['alphac']):.3f}")
    ax.axvline(np.median(lj), color="C0", lw=2, ls="--",
               label=f"median = {np.median(lj):.3f}")
    kde_J = gaussian_kde(lj); gJ = np.linspace(lj.min(), lj.max(), 2001)
    MAP_J = float(gJ[np.argmax(kde_J(gJ))])
    ax.axvline(MAP_J, color="C2", lw=2, ls=":", label=f"1D MAP = {MAP_J:.3f}")
    ax.set_xlabel(r"$\log_{10} J(\alpha_c)\ [M_\odot^2 / kpc^5]$")
    ax.set_ylabel("density")
    ax.set_title("(c) log J(α_c) — MAP at truth, median offset")
    ax.legend(fontsize=8, frameon=False)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "asimov_diagnostic.png", dpi=130)
    print(f"saved triptych figure to {OUT_DIR / 'asimov_diagnostic.png'}")
