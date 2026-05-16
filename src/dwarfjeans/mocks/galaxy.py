"""
Mock dwarf-galaxy generator for Monte Carlo Jeans-recovery tests.

Generates per-star (R_i, V_i, sigma_eps_i, p_i) catalogs for a galaxy with:
  * Plummer tracer with scale radius r_p
  * NFW halo (r_s, rho_s)
  * Constant anisotropy beta (default isotropic)

Steps:
  1. Sample N projected radii R_i from the 2D Plummer surface density.
     The 2D Plummer CDF is F(<R) = R^2 / (R^2 + r_p^2), so F^-1(u) = r_p * sqrt(u/(1-u)).
  2. For each R_i, evaluate sigma_los(R_i) from the spherical Jeans projection
     (this module's `jeans.sigma_los`).
  3. Draw V_i ~ Normal(V_systemic, sqrt(sigma_los(R_i)^2 + sigma_eps_i^2)).
  4. Set p_i = 1 (post-membership-cut convention; matches our likelihood).

The output schema matches the per-star spectroscopic schema documented in
pipeline_overview.md / data_sources.md: (R_i [kpc], V_i [km/s],
sigma_eps_i [km/s], p_i).
"""

from __future__ import annotations

import numpy as np

from dwarfjeans.jeans import solver as jeans


def beta_tilde_to_beta(beta_tilde: float) -> float:
    """Symmetrized anisotropy: beta = 2 beta_tilde / (1 + beta_tilde)."""
    return 2.0 * beta_tilde / (1.0 + beta_tilde)


def beta_to_beta_tilde(beta: float) -> float:
    """Inverse: beta_tilde = beta / (2 - beta)."""
    return beta / (2.0 - beta)


def sample_plummer_R(n: int, r_p: float, rng: np.random.Generator) -> np.ndarray:
    """
    Draw n projected radii R_i from the 2D Plummer surface density.

    Sigma(R) ∝ (1 + R^2/r_p^2)^-2  →  F(<R) = R^2 / (R^2 + r_p^2)
    Inverse-CDF: R = r_p * sqrt(u / (1 - u)),  u ~ U(0, 1).
    """
    u = rng.uniform(0.0, 1.0, size=n)
    return r_p * np.sqrt(u / (1.0 - u))


def plummer_R_stratified(n: int, r_p: float, R_max: float) -> np.ndarray:
    """
    Deterministic equal-probability-stratified midpoint sample of n radii
    from the *truncated* 2D Plummer distribution on R ∈ [0, R_max].

    The truncated CDF is F_t(R) = F(R) / F(R_max), where F(R) = R²/(R²+r_p²).
    Stratified midpoints u_i = (i - 0.5)/n, i = 1..n in stratum-probability
    space are inverted via R = r_p · sqrt(u·F_max / (1 - u·F_max)).

    Reproduces the truncated Plummer moments exactly to O(1/n²) for smooth
    integrands and is symmetric in i (no random component). This is the
    Asimov-realization R sample.
    """
    F_max = R_max ** 2 / (R_max ** 2 + r_p ** 2)
    u = (np.arange(n) + 0.5) / n
    v = u * F_max
    return r_p * np.sqrt(v / (1.0 - v))


def make_mock_galaxy(
    *,
    n_stars: int,
    r_s: float,           # kpc
    rho_s: float,         # M_sun / kpc^3
    r_p: float,           # kpc
    beta: float = 0.0,
    V_sys: float = 0.0,   # km/s
    sigma_eps: float | np.ndarray = 2.0,  # km/s; scalar or per-star
    R_max_factor: float = 5.0,  # truncate sampling at R_max = R_max_factor * r_p
    rng: np.random.Generator | None = None,
    # Optional perspective-motion injection.
    d_kpc: float | None = None,
    ra_center_deg: float | None = None,
    dec_center_deg: float | None = None,
    pmra_true_mas_yr: float | None = None,
    pmdec_true_mas_yr: float | None = None,
) -> dict:
    """
    Generate a mock galaxy and return a dict with:
      R, V, sigma_eps, p           — per-star arrays (kpc, km/s, km/s, dimensionless)
      sigma_los_true               — per-star evaluated sigma_los(R_i) [km/s]
      truth                        — dict of true parameters

    R_max_factor: enforce R_i <= R_max_factor * r_p by rejecting + redrawing,
    keeping the sample within a regime where Plummer-modeled membership
    cuts in real catalogs would also have already pruned outliers.

    **Perspective-motion injection (optional).** When all five of
    ``d_kpc``, ``ra_center_deg``, ``dec_center_deg``, ``pmra_true_mas_yr``,
    ``pmdec_true_mas_yr`` are provided, the mock additionally:
      * generates per-star (RA, Dec) by drawing azimuth uniformly and
        converting the projected R to a tangent-plane offset (Δα·cos δ_0,
        Δδ) at the given galaxy center;
      * adds a true Kaplinghat–Strigari shift
        Δv_persp = A·d·(μ_α*·Δα·cos δ_0 + μ_δ·Δδ) to each V_i, with the
        Gaussian dispersion noise still on σ_los(R_i)² + σ_eps².
    The output dict then also carries ``RA_star``, ``Dec_star``,
    ``V_observed`` (= V), ``dv_persp_true``, plus PM truth keys.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Step 1: positions. Draw with rejection so all R_i <= R_max.
    R_max = R_max_factor * r_p
    R = np.empty(0)
    while R.size < n_stars:
        block = sample_plummer_R(n_stars * 2, r_p, rng)
        R = np.concatenate([R, block[block <= R_max]])
    R = R[:n_stars]

    # Step 2: Jeans-projected sigma_los at each R_i (slow, but only once).
    sigma_los_true = jeans.sigma_los(R, beta, r_s, rho_s, r_p, method="grid")

    # Step 3: per-star velocity errors.
    if np.isscalar(sigma_eps):
        sigma_eps_arr = np.full(n_stars, float(sigma_eps))
    else:
        sigma_eps_arr = np.asarray(sigma_eps, dtype=float)
        assert sigma_eps_arr.shape == (n_stars,), "sigma_eps shape mismatch"

    # Step 4: V_i ~ Normal(V_sys, sqrt(sigma_los^2 + sigma_eps^2)).
    total_sigma = np.sqrt(sigma_los_true ** 2 + sigma_eps_arr ** 2)
    V = V_sys + rng.normal(0.0, total_sigma)

    # Step 4b: optional perspective injection.
    persp_inputs = (d_kpc, ra_center_deg, dec_center_deg,
                    pmra_true_mas_yr, pmdec_true_mas_yr)
    has_perspective = all(v is not None for v in persp_inputs)
    if any(v is not None for v in persp_inputs) and not has_perspective:
        raise ValueError(
            "perspective injection requires all of d_kpc, ra_center_deg, "
            "dec_center_deg, pmra_true_mas_yr, pmdec_true_mas_yr"
        )
    if has_perspective:
        from dwarfjeans.jeans.perspective import (
            A_KMS_PER_MASYR_KPC, perspective_correction,
        )
        # Per-star angular offsets: pick uniform azimuth, set radial offset
        # so that the projected R matches the Plummer draw exactly.
        theta = rng.uniform(0.0, 2.0 * np.pi, size=n_stars)
        rho_rad = R / float(d_kpc)              # small-angle: ρ = R/d
        cos_d0 = float(np.cos(np.deg2rad(dec_center_deg)))
        dRA_deg = np.rad2deg(rho_rad * np.cos(theta)) / cos_d0
        dDec_deg = np.rad2deg(rho_rad * np.sin(theta))
        RA_star = ra_center_deg + dRA_deg
        Dec_star = dec_center_deg + dDec_deg
        dv_persp_true = perspective_correction(
            ra_deg=RA_star, dec_deg=Dec_star,
            ra_center_deg=ra_center_deg, dec_center_deg=dec_center_deg,
            distance_kpc=d_kpc,
            pm_alpha_star_masyr=pmra_true_mas_yr,
            pm_delta_masyr=pmdec_true_mas_yr,
        )
        V_observed = V + dv_persp_true
    else:
        RA_star = Dec_star = dv_persp_true = None
        V_observed = V

    p = np.ones(n_stars)

    # Plummer half-light radii. The 2D-projected half-light radius equals
    # r_p exactly for a spherical Plummer surface density (CDF F(<R) =
    # R²/(R²+r_p²) ⇒ R_half = r_p). The 3D half-mass radius is the
    # Plummer-specific factor 1.30477·r_p (numerical root of
    # r³/(r²+r_p²)^(3/2) = 1/2). Wolf+2010's mass estimator uses the
    # 3D radius; the pipeline docs use 2D for r_1/2.
    R_half_2d = r_p                      # kpc
    r_half_3d = 1.30476740610256 * r_p   # kpc
    M_half_2d = float(jeans.nfw_M(R_half_2d, r_s, rho_s))
    M_half_3d = float(jeans.nfw_M(r_half_3d, r_s, rho_s))

    truth = {
        "n_stars": n_stars,
        "r_s": r_s,
        "rho_s": rho_s,
        "r_p": r_p,
        "beta": beta,
        "beta_tilde": beta_to_beta_tilde(beta),
        "V_sys": V_sys,
        "log10_rs": np.log10(r_s),
        "log10_rhos": np.log10(rho_s),
        "log10_rhos_rs3": np.log10(rho_s * r_s ** 3),
        "R_half_2d": R_half_2d,
        "r_half_3d": r_half_3d,
        "M_half_2d": M_half_2d,
        "M_half_3d": M_half_3d,
        "log10_M_half_2d": float(np.log10(M_half_2d)),
        "log10_M_half_3d": float(np.log10(M_half_3d)),
    }
    if has_perspective:
        truth.update({
            "d_kpc":      float(d_kpc),
            "ra_center":  float(ra_center_deg),
            "dec_center": float(dec_center_deg),
            "pmra":       float(pmra_true_mas_yr),
            "pmdec":      float(pmdec_true_mas_yr),
        })

    out = {
        "R": R,
        "V": V,
        "sigma_eps": sigma_eps_arr,
        "p": p,
        "sigma_los_true": sigma_los_true,
        "truth": truth,
    }
    if has_perspective:
        out["RA_star"] = RA_star
        out["Dec_star"] = Dec_star
        out["V_observed"] = V_observed
        out["dv_persp_true"] = dv_persp_true
    return out


def make_asimov_galaxy(
    *,
    n_stars: int,
    r_s: float,           # kpc
    rho_s: float,         # M_sun / kpc^3
    r_p: float,           # kpc
    beta: float = 0.0,
    V_sys: float = 0.0,   # km/s — placeholder, not used by the Asimov likelihood
    sigma_eps: float | np.ndarray = 2.0,  # km/s
    R_max_factor: float = 5.0,
) -> dict:
    """
    Generate a single Asimov 'realization' for fast dev-loop validation.

    Construction:
      * R_i — deterministic equal-probability-stratified midpoints of the
        2D Plummer surface density truncated at R_max = R_max_factor · r_p
        (matches the rejection cap in `make_mock_galaxy`).
      * sigma_los_truth(R_i) — Jeans-projected at the truth (r_s, ρ_s, β, r_p).
      * sigma_eps_i — scalar or per-star.
      * V_i — placeholder, set to V_sys.
      * p_i = 1.

    The Asimov likelihood (`make_loglike_asimov` in dwarfjeans.jeans.inference) replaces
    each (V_i - V_sys)² in the Gaussian likelihood with its expectation value
    σ_tot,truth²(R_i) = sigma_los_truth(R_i)² + sigma_eps_i², so the V_i array
    is *not consumed* by the Asimov inference. We populate V = V_sys so the
    on-disk schema matches the MC mocks.

    A consequence: V_sys is unconstrained by the Asimov likelihood and its
    posterior is the prior. Asimov summaries should flag V as 'prior-only'
    rather than reporting a misleading z-score. The halo parameters are
    constrained as in the MC case.

    Returns the same dict schema as `make_mock_galaxy`, plus `is_asimov: True`.
    """
    R_max = R_max_factor * r_p
    R = plummer_R_stratified(n_stars, r_p, R_max)
    sigma_los_truth = jeans.sigma_los(R, beta, r_s, rho_s, r_p, method="grid")

    if np.isscalar(sigma_eps):
        sigma_eps_arr = np.full(n_stars, float(sigma_eps))
    else:
        sigma_eps_arr = np.asarray(sigma_eps, dtype=float)
        assert sigma_eps_arr.shape == (n_stars,), "sigma_eps shape mismatch"

    V = np.full(n_stars, float(V_sys))   # placeholder, not used by Asimov loglike
    p = np.ones(n_stars)

    R_half_2d = r_p
    r_half_3d = 1.30476740610256 * r_p
    M_half_2d = float(jeans.nfw_M(R_half_2d, r_s, rho_s))
    M_half_3d = float(jeans.nfw_M(r_half_3d, r_s, rho_s))

    truth = {
        "n_stars": n_stars,
        "r_s": r_s,
        "rho_s": rho_s,
        "r_p": r_p,
        "beta": beta,
        "beta_tilde": beta_to_beta_tilde(beta),
        "V_sys": V_sys,
        "log10_rs": np.log10(r_s),
        "log10_rhos": np.log10(rho_s),
        "log10_rhos_rs3": np.log10(rho_s * r_s ** 3),
        "R_half_2d": R_half_2d,
        "r_half_3d": r_half_3d,
        "M_half_2d": M_half_2d,
        "M_half_3d": M_half_3d,
        "log10_M_half_2d": float(np.log10(M_half_2d)),
        "log10_M_half_3d": float(np.log10(M_half_3d)),
    }

    return {
        "R": R,
        "V": V,
        "sigma_eps": sigma_eps_arr,
        "p": p,
        "sigma_los_true": sigma_los_truth,
        "truth": truth,
        "is_asimov": True,
    }


if __name__ == "__main__":
    # Sanity check: make a small mock galaxy and print summary statistics.
    rng = np.random.default_rng(0)
    g = make_mock_galaxy(
        n_stars=200,
        r_s=1.0, rho_s=1e7, r_p=0.30,
        beta=0.0, V_sys=0.0, sigma_eps=2.0,
        rng=rng,
    )
    print(f"n_stars = {len(g['R'])}")
    print(f"R: median {np.median(g['R']):.4f} kpc, max {g['R'].max():.4f} kpc")
    print(f"V: mean {g['V'].mean():.3f} km/s, std {g['V'].std():.3f} km/s")
    print(f"sigma_los_true: median {np.median(g['sigma_los_true']):.3f} km/s, "
          f"min {g['sigma_los_true'].min():.3f}, max {g['sigma_los_true'].max():.3f}")
    print(f"truth: {g['truth']}")
