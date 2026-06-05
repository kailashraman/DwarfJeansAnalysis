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
from numpy.polynomial.legendre import leggauss

from dwarfjeans.jeans import solver as jeans  # only for the gravitational constant check + sanity


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
# Exact-geometry J-factor and angular containment
#
# Ported from SatGen_Dwarf/python/Jdwarf.py:compute_J_and_containment, which is
# validated to <0.1% vs scipy dblquad. Unlike J_D_factors (small-angle, R = d·θ)
# this uses the exact line-of-sight geometry (law of cosines, half-angle form)
# valid to θ = π/2, needed because the 95% containment angle reaches well beyond
# the small-angle regime for nearby dwarfs. The halo is the NFW truncated at the
# tidal radius r_t, so the "full" J is J integrated out to r_t.
# ----------------------------------------------------------------------------

def los_radius(L, d, theta):
    """3D distance from the dwarf centre to a point at line-of-sight distance L
    and angle theta from the dwarf direction.

    Algebraically the law of cosines L² + d² − 2 L d cos θ, but written in the
    half-angle form (L−d)² + 4 L d sin²(θ/2), which is identical and manifestly
    ≥ 0 (no catastrophic cancellation for L≈d, θ≈0). Floored just off zero so the
    central r=0 NFW cusp (ρ ~ 1/r, integrable) does not divide by zero.
    """
    return np.maximum(np.sqrt((L - d) ** 2 + 4.0 * L * d * np.sin(theta / 2.0) ** 2),
                      1e-6)


def _truncated_nfw_rho(r_s, rho_s, r_t):
    """NFW density truncated hard at r_t (zero beyond), as a callable of r."""
    def rho(r):
        r = np.asarray(r, dtype=float)
        return np.where(r <= r_t, nfw_rho(r, r_s, rho_s), 0.0)
    return rho


def j_containment(rho, d_kpc, r_t, aperture_rad, frac=0.95,
                  n_theta=400, n_L=64):
    """Exact-geometry J(<aperture) and the `frac` containment angle.

    `rho` is the (truncated) 3D density callable [Msun/kpc³]. Integrates the
    angular J-profile dJ/dθ out to the angle subtended by r_t (arcsin(r_t/d)),
    so the "full" J is J within the tidal radius. Returns

        (J_aperture, theta_frac)

    with J_aperture in GeV²/cm⁵ (the module LOG10_J_FAC conversion) and
    theta_frac in radians (the angle containing `frac` of the full J).

    Mirrors SatGen_Dwarf compute_J_and_containment: deterministic Gauss-Legendre
    line-of-sight quadrature with the closest-approach substitution
    L − L0 = rmin tan φ (L0 = d cos θ, rmin = d sin θ), which clusters nodes where
    ρ peaks. Returns (nan, nan) if r_t or d is non-positive.
    """
    d = float(d_kpc)
    if not (d > 0.0) or not (r_t > 0.0):
        return float("nan"), float("nan")

    # Max angle the truncated halo subtends: a LOS at impact parameter
    # rmin = d sin θ enters the r_t sphere iff sin θ ≤ r_t/d.
    theta_max = float(np.arcsin(min(r_t / d, 1.0)))
    aperture = float(aperture_rad)
    if not (theta_max > 0.0):
        return float("nan"), float("nan")
    # If the aperture exceeds the halo's angular extent, J(<aperture) = J_full.
    aperture_eff = min(aperture, theta_max)

    # log-spaced theta grid, with theta=0 and the aperture forced in as exact
    # nodes so J(<aperture) is read off directly.
    th = np.unique(np.concatenate([
        [0.0],
        np.logspace(np.log10(aperture_eff * 1e-3), np.log10(theta_max), n_theta),
        [aperture_eff]]))

    # I(theta) = int_0^{2d} rho(r)^2 dL via Gauss-Legendre with the substitution
    # L - L0 = rmin tan(phi): r = rmin sec(phi), dL = rmin sec^2(phi) dphi.
    gx, gw = leggauss(n_L)
    L0 = d * np.cos(th)
    rmin = np.maximum(d * np.sin(th), 1e-12)
    phi_lo = np.arctan((0.0 - L0) / rmin)
    phi_hi = np.arctan((2 * d - L0) / rmin)
    half = 0.5 * (phi_hi - phi_lo)
    mid = 0.5 * (phi_hi + phi_lo)
    phi = mid[:, None] + half[:, None] * gx[None, :]
    r = rmin[:, None] / np.cos(phi)
    integrand = rho(r) ** 2 * (rmin[:, None] / np.cos(phi) ** 2)
    I = (integrand * (half[:, None] * gw[None, :])).sum(axis=1)

    w = 2 * np.pi * np.sin(th) * I               # dJ/dtheta
    # At theta=0 the cusp makes I ~ 1/rmin diverge while sin(theta) -> 0; the
    # product (dJ/dtheta) tends to a finite NONZERO constant, but the literal
    # w[0] = 2pi*sin(0)*I(0) collapses 0*inf to 0 and under-counts the first
    # interval. Use the cusp limit. Valid for an NFW gamma=1 inner cusp
    # (dJ/dtheta ~ theta^{2-2gamma} -> const); a steeper cusp would need
    # analytic treatment of the divergent first interval.
    w[0] = w[1]
    cdf = np.concatenate([[0.0],
                          np.cumsum(0.5 * (w[1:] + w[:-1]) * np.diff(th))])

    J_full = cdf[-1]
    J_aperture = float(np.interp(aperture_eff, th, cdf))   # Msun²/kpc⁵
    theta_frac = float(np.interp(frac * J_full, cdf, th))
    # Convert J(<aperture) to GeV²/cm⁵ (same factor as the rest of the module).
    J_aperture_gev = (J_aperture * 10.0 ** LOG10_J_FAC) if J_aperture > 0 else float("nan")
    return J_aperture_gev, theta_frac


def j_aperture_and_containment(d_kpc, r_s, rho_s, r_t, aperture_rad=0.5 * np.pi / 180.0,
                               frac=0.95, n_theta=400, n_L=64):
    """Convenience wrapper: build the r_t-truncated NFW and call j_containment.

    Returns (J_aperture [GeV²/cm⁵], theta_frac [rad]).
    """
    rho = _truncated_nfw_rho(r_s, rho_s, r_t)
    return j_containment(rho, d_kpc, r_t, aperture_rad, frac=frac,
                         n_theta=n_theta, n_L=n_L)


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
