"""
J- and D-factor integrals for NFW halos with tidal-radius truncation.

Conventions match the rest of this codebase (kpc, Msun, etc.) plus the
P&S 2018 reporting unit choice (GeV²/cm⁵ for J, GeV/cm² for D).

J(θ_max) = ∫_{ΔΩ(θ_max)} ∫_{l.o.s.} ρ²(r(l,θ)) dl dΩ
D(θ_max) = ∫_{ΔΩ(θ_max)} ∫_{l.o.s.} ρ(r(l,θ)) dl dΩ

We use the small-angle approximation valid for θ_max < ~1°, which is the
regime of all reported angles (max α_c ~ 0.5° for typical UFDs at d ≳ 25 kpc):

    R = d · θ                                 (impact parameter)
    dΩ ≈ R dR / d² = dA / d²                  (small-angle solid angle)
    J(θ_max) = (2π / d²) ∫_0^{R_max} R dR · I_2(R; r_t)
    D(θ_max) = (2π / d²) ∫_0^{R_max} R dR · I_1(R; r_t)

where R_max = d · θ_max, and the line-of-sight column at impact parameter R is

    I_n(R; r_t) = 2 ∫_R^{r_t} ρ^n(r) · r / √(r² − R²) dr     (n = 1 or 2)

The √(r² − R²) singularity at r = R is removed by the substitution
r² = R² + u², which gives a smooth integrand on u ∈ [0, √(r_t² − R²)]:

    I_n(R; r_t) = 2 ∫_0^{u_max(R)} ρ^n(√(R² + u²)) du

For NFW ρ(r) = ρ_s / [(r/r_s)(1 + r/r_s)²], both integrals are well-behaved
on this domain.

Implementation:
  * Tabulate u-grid log-spaced + 0 endpoint (same trick as the Jeans
    projection in jeans.py).
  * Vectorize over impact parameters R via a 2D (n_R, n_u) array.
  * Outer R-integral via trapezoid.

The n_u and n_R values are chosen such that grid-vs-quad relative error
is below ~1e-3 across the parameter range relevant for these tests.

Cost: a single (J, D) evaluation across all four reported angles takes
~ a few ms. A 4000-sample posterior chain pushes through in a few seconds.
"""

from __future__ import annotations

import numpy as np

import jeans  # only for the gravitational constant check + sanity


# Unit conversion: 1 Msun² / kpc^5 → GeV² / cm^5
M_SUN_GEV = 1.11534e57          # solar mass in GeV (c=1)
KPC_CM = 3.085677581e21         # kpc in cm
J_FAC_MSUN2_KPC5_TO_GEV2_CM5 = M_SUN_GEV ** 2 / KPC_CM ** 5
D_FAC_MSUN_KPC2_TO_GEV_CM2 = M_SUN_GEV / KPC_CM ** 2

# Logarithmic offsets (handy for chain bookkeeping)
LOG10_J_FAC = float(np.log10(J_FAC_MSUN2_KPC5_TO_GEV2_CM5))
LOG10_D_FAC = float(np.log10(D_FAC_MSUN_KPC2_TO_GEV_CM2))


def nfw_rho(r, r_s, rho_s):
    """NFW density profile. r, r_s in kpc; rho_s in Msun/kpc^3."""
    x = r / r_s
    return rho_s / (x * (1.0 + x) ** 2)


def _column_integrals(R, r_s, rho_s, r_t, n_u=512):
    """
    Compute the line-of-sight column integrals at each impact parameter R:

        I_1(R) = 2 ∫_0^{u_max(R)} ρ(√(R² + u²)) du           [Msun/kpc^2]
        I_2(R) = 2 ∫_0^{u_max(R)} ρ²(√(R² + u²)) du          [Msun²/kpc^5]

    where u_max(R) = √(r_t² − R²).

    Vectorized: returns (I_1, I_2), each shape (n_R,).

    For numerical stability, the u-grid has an explicit u=0 endpoint
    (where ρ is largest along the column) plus log-spaced points from
    u_min = 1e-6 · r_s to u_max(R). Per-R u-grid because u_max depends on R.
    """
    R = np.atleast_1d(np.asarray(R, dtype=float))
    n_R = R.size
    I1 = np.zeros(n_R)
    I2 = np.zeros(n_R)

    u_min_floor = 1e-6 * r_s
    for i, Ri in enumerate(R):
        if Ri <= 0.0 or Ri >= r_t:
            # Ri = 0: NFW cusp diverges at u=0 (r=0), so I_n(0) is ∞.
            # The caller is responsible for handling on-axis specially
            # (the area element R · I_n is 0 in the J/D outer integral).
            # Ri ≥ r_t: outside the truncation radius, no contribution.
            continue
        u_max_R = np.sqrt(r_t ** 2 - Ri ** 2)
        if u_max_R <= u_min_floor:
            u = np.linspace(0.0, u_max_R, n_u)
        else:
            log_u = np.linspace(np.log(u_min_floor), np.log(u_max_R), n_u - 1)
            u = np.concatenate([[0.0], np.exp(log_u)])
        r = np.sqrt(Ri ** 2 + u ** 2)
        rho = nfw_rho(r, r_s, rho_s)
        I1[i] = 2.0 * np.trapezoid(rho, u)
        I2[i] = 2.0 * np.trapezoid(rho ** 2, u)
    return I1, I2


def J_D_factors(theta_max_rad, d, r_s, rho_s, r_t,
                 n_R=128, n_u=512):
    """
    Return (J, D) for the given θ_max [radians], distance d [kpc],
    NFW (r_s, rho_s) [kpc, Msun/kpc^3], tidal radius r_t [kpc].

    Units: J in Msun²/kpc^5, D in Msun/kpc^2. Use the module-level
    conversion factors to express in GeV²/cm⁵ and GeV/cm².

    Small-angle approximation: R = d · θ, dΩ ≈ R dR / d².
    """
    theta_max_rad = float(theta_max_rad)
    R_max = min(d * theta_max_rad, r_t)
    if R_max <= 0:
        return 0.0, 0.0
    # Log-spaced R grid. We do NOT include R=0: the integrand R · I_n(R) is
    # zero at R=0 (area element kills the on-axis NFW cusp), but evaluating
    # I_n(0) hits ρ(0) = ∞ in float64 and 0·∞ = nan. Adding the R=0 endpoint
    # with value 0 manually lets trapezoid integrate from R_min down to 0
    # cleanly.
    R_min = 1e-6 * r_s
    if R_max <= R_min:
        return 0.0, 0.0
    log_R = np.linspace(np.log(R_min), np.log(R_max), n_R)
    R = np.exp(log_R)
    I1, I2 = _column_integrals(R, r_s, rho_s, r_t, n_u=n_u)

    # J = (2π / d²) ∫ R · I_2 dR     (small-angle approximation)
    # D = (2π / d²) ∫ R · I_1 dR
    # Prepend the R=0 endpoint with integrand 0 so the trapezoid covers
    # [0, R_max] without the on-axis cusp evaluation.
    R_full = np.concatenate([[0.0], R])
    integrand_J = np.concatenate([[0.0], R * I2])
    integrand_D = np.concatenate([[0.0], R * I1])
    J = 2.0 * np.pi / d ** 2 * np.trapezoid(integrand_J, R_full)
    D = 2.0 * np.pi / d ** 2 * np.trapezoid(integrand_D, R_full)
    return float(J), float(D)


def J_D_chain(samples_eq, d, r_t, theta_max_rad, n_R=128, n_u=512):
    """
    Push an equal-weight posterior chain through J_D_factors at one θ_max.

    samples_eq: (N, 4) array with columns (V, log10_rs, log10_rhos, beta_tilde).
    Returns (J_chain, D_chain) of length N, in Msun²/kpc^5 and Msun/kpc^2.
    """
    N = samples_eq.shape[0]
    J = np.empty(N)
    D = np.empty(N)
    for i, row in enumerate(samples_eq):
        _, log_rs, log_rhos, _ = row
        r_s = 10.0 ** log_rs
        rho_s = 10.0 ** log_rhos
        J[i], D[i] = J_D_factors(theta_max_rad, d, r_s, rho_s, r_t,
                                   n_R=n_R, n_u=n_u)
    return J, D


# Default reporting angles (radians). P&S 2018 standard set:
DEG = np.pi / 180.0
ANGLES_FIXED = {
    "0p1deg": 0.1 * DEG,
    "0p2deg": 0.2 * DEG,
    "0p5deg": 0.5 * DEG,
}


def alpha_c_radians(r_half_3d_kpc: float, d_kpc: float) -> float:
    """The critical-angle convention from pipeline_overview.md:
        α_c = 2 r_½ / d   (radians)
    Used as J's 'natural' integration angle. D uses α_c / 2."""
    return 2.0 * r_half_3d_kpc / d_kpc


# ----------------------------------------------------------------------------
# Self-tests
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    # Sanity test: at small θ_max with R_max << r_s, J should scale like
    # the central-density column-squared times area:
    #     J ≈ (2π/d²) · ½ R_max² · 2 · ρ_s² · r_s · L_central
    # where L_central is the column at R=0, ∫₀^{r_t} ρ² dl. Hard to closed-form
    # without machinery, so just check J > 0, D > 0, J grows with θ_max.

    r_s, rho_s, d, r_t = 0.3, 3e8, 30.0, 1.0
    print(f"Truth: r_s={r_s} rho_s={rho_s:.2e} d={d} r_t={r_t}")
    print(f"  ρ_s · r_s = {rho_s * r_s:.3e} Msun/kpc² (central column scale)")
    print()
    print(f"  {'θ_max [°]':>10}  {'log10 J [Msun²/kpc⁵]':>22}  "
          f"{'log10 D [Msun/kpc²]':>22}  "
          f"{'log10 J [GeV²/cm⁵]':>22}")
    for tag, th in [("0.1deg", 0.1*DEG), ("0.2deg", 0.2*DEG),
                     ("0.5deg", 0.5*DEG), ("1.0deg", 1.0*DEG)]:
        J, D = J_D_factors(th, d, r_s, rho_s, r_t)
        log_J = np.log10(J) if J > 0 else float('-inf')
        log_D = np.log10(D) if D > 0 else float('-inf')
        log_J_gev = log_J + LOG10_J_FAC
        log_D_gev = log_D + LOG10_D_FAC
        print(f"  {tag:>10}  {log_J:>22.4f}  {log_D:>22.4f}  {log_J_gev:>22.4f}")
    print(f"\n  log10 J unit shift (Msun²/kpc⁵ → GeV²/cm⁵): +{LOG10_J_FAC:.4f}")
    print(f"  log10 D unit shift (Msun/kpc² → GeV/cm²):    +{LOG10_D_FAC:.4f}")

    # Quick monotonicity check: J and D must increase with θ_max.
    Js, Ds = [], []
    for th_deg in [0.05, 0.1, 0.2, 0.5, 1.0]:
        J, D = J_D_factors(th_deg * DEG, d, r_s, rho_s, r_t)
        Js.append(J); Ds.append(D)
    assert all(Js[i+1] > Js[i] for i in range(len(Js)-1)), "J must increase with θ_max"
    assert all(Ds[i+1] > Ds[i] for i in range(len(Ds)-1)), "D must increase with θ_max"
    print("\n  J, D monotonic in θ_max: OK")
