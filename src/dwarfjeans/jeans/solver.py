"""
Bare-bones spherical Jeans projection: NFW halo, Plummer tracer, constant β.

Conventions match Pace & Strigari 2018:
- r_s, r_p, r_1/2, R    in kpc internally (caller responsibility)
- rho_s                  in M_sun / kpc^3
- sigma_los, sigma_r     in km/s
- M(r)                   in M_sun
- G                      in (km/s)^2 kpc / M_sun  (so v^2 = G M / r works in km/s)

We implement P&S eq. 3-7. The two integrals that have to be done numerically:

  Jeans (eq. 3, constant-β closed-form integrating-factor solution):
    nu(r) * sigma_r^2(r) = r^(-2β) * integral from r to infinity of
                            s^(2β) * nu(s) * G * M(s) / s^2 ds        (J1)

  Projection (eq. 4):
    Sigma(R) * sigma_los^2(R) = 2 * integral from R to infinity of
                                 (1 - β R^2 / r^2) * r * nu(r) * sigma_r^2(r) / sqrt(r^2 - R^2) dr  (P1)

We handle the sqrt(r^2 - R^2) singularity in (P1) via substitution r^2 = R^2 + u^2,
giving a smooth integrand on u in [0, inf).

This module is a SKELETON — to be iterated on. Two implementations of each
integral are provided side by side: a fast vectorized trapezoidal version
on a tabulated grid, and a slow trusted scipy.integrate.quad reference.
The test script cross-checks them.
"""

from __future__ import annotations

import numpy as np
from scipy import integrate

# Gravitational constant in units of (km/s)^2 * kpc / M_sun.
# G_SI = 6.6743e-11 m^3 kg^-1 s^-2
# Convert: kpc = 3.0857e19 m, M_sun = 1.989e30 kg, km/s = 1e3 m/s
# G [km^2/s^2 kpc / M_sun] = G_SI * M_sun / kpc / (km/s)^2
#                          = 6.6743e-11 * 1.989e30 / 3.0857e19 / 1e6
G_KPC_KMS2_MSUN = 4.3009e-6  # standard value


# ----------------------------------------------------------------------------
# NFW enclosed mass with small-x series
# ----------------------------------------------------------------------------

def nfw_g(x: np.ndarray | float) -> np.ndarray | float:
    """
    Returns g(x) = ln(1 + x) - x/(1 + x), the NFW dimensionless enclosed-mass
    function. Uses a Taylor series for small x to avoid catastrophic cancellation
    in ln(1+x) - x/(1+x).

    g(x) ≈ x^2/2 - 2x^3/3 + 3x^4/4 - 4x^5/5 + ...     (alternating series)
    The first few terms suffice well below x ~ 1e-3.
    """
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)

    small = x < 1e-3
    big = ~small

    # Direct form for x not small
    if np.any(big):
        xb = x[big]
        out[big] = np.log1p(xb) - xb / (1.0 + xb)

    # Taylor series for small x: keep through x^6 (relative error << 1e-15 at x=1e-3)
    if np.any(small):
        xs = x[small]
        x2 = xs * xs
        x3 = x2 * xs
        x4 = x2 * x2
        x5 = x4 * xs
        x6 = x4 * x2
        out[small] = (
            0.5 * x2
            - (2.0 / 3.0) * x3
            + 0.75 * x4
            - 0.8 * x5
            + (5.0 / 6.0) * x6
        )

    return out if out.shape else float(out)


def nfw_M(r: np.ndarray | float, r_s: float, rho_s: float) -> np.ndarray | float:
    """NFW enclosed mass M(<r) = 4π ρ_s r_s^3 g(r/r_s)."""
    return 4.0 * np.pi * rho_s * r_s ** 3 * nfw_g(np.asarray(r) / r_s)


def nfw_h(x: np.ndarray | float) -> np.ndarray | float:
    """
    Auxiliary dimensionless mass function h(x) = x^2 / (1+x)^2 that arises
    from -∂g/∂ln r_s at fixed r' (with x = r'/r_s). Used to construct the
    Fisher-information shape factor T = 3 - 𝒬/𝒫 for the conditional
    Jeffreys prior on (ln ρ_s, ln r_s) — see jeffreys_jeans_derivation.md.
    """
    x = np.asarray(x, dtype=float)
    return x * x / (1.0 + x) ** 2


# ----------------------------------------------------------------------------
# Plummer tracer density
# ----------------------------------------------------------------------------

def plummer_nu(r: np.ndarray | float, r_p: float) -> np.ndarray | float:
    """3D Plummer tracer number density. Normalization 3/(4π r_p^3)."""
    r = np.asarray(r, dtype=float)
    return (3.0 / (4.0 * np.pi * r_p ** 3)) * (1.0 + (r / r_p) ** 2) ** (-2.5)


def plummer_Sigma(R: np.ndarray | float, r_p: float) -> np.ndarray | float:
    """2D projected Plummer surface density. Normalization 1/(π r_p^2)."""
    R = np.asarray(R, dtype=float)
    return (1.0 / (np.pi * r_p ** 2)) * (1.0 + (R / r_p) ** 2) ** (-2)


# ----------------------------------------------------------------------------
# Inner Jeans integral (J1): nu(r) sigma_r^2(r)
# Vectorized tabulated version + scipy reference
# ----------------------------------------------------------------------------

def _jeans_integrand(s, beta, r_s, rho_s, r_p):
    """Integrand for J1: s^(2β) * ν(s) * G M(s) / s^2."""
    return s ** (2.0 * beta) * plummer_nu(s, r_p) * G_KPC_KMS2_MSUN * nfw_M(s, r_s, rho_s) / s ** 2


def nu_sigma_r2_quad(
    r: np.ndarray,
    beta: float,
    r_s: float,
    rho_s: float,
    r_p: float,
    s_inf: float | None = None,
) -> np.ndarray:
    """
    Reference (slow, trusted) scipy.integrate.quad evaluation of ν(r) σ_r²(r).
    Returns an array of the same shape as r.

    s_inf sets the effective upper limit; default is 100 * max(r_s, r_p),
    which is well into the Plummer r^-5 tail and the NFW logarithmic regime.
    """
    if s_inf is None:
        s_inf = 100.0 * max(r_s, r_p)

    r = np.atleast_1d(np.asarray(r, dtype=float))
    out = np.empty_like(r)
    for i, ri in enumerate(r):
        val, _ = integrate.quad(
            _jeans_integrand, ri, s_inf,
            args=(beta, r_s, rho_s, r_p),
            limit=200,
            epsabs=0.0, epsrel=1e-9,
        )
        out[i] = ri ** (-2.0 * beta) * val
    return out


def nu_sigma_r2_grid(
    r: np.ndarray,
    beta: float,
    r_s: float,
    rho_s: float,
    r_p: float,
    n_grid: int = 4096,
    s_inf: float | None = None,
) -> np.ndarray:
    """
    Vectorized tabulated evaluation of ν(r) σ_r²(r) using a single shared
    log-spaced grid in s. Cumulative-from-the-right trapezoidal integration
    in log-space, to avoid float64 overflow at extreme β.

    Returns an array of the same shape as r.

    Why log-space. The integrand contains s^(2β); at the tangential prior
    edge β̃ = -0.95 ⇒ β = -38, so 2β = -76. With s_min ~ 3e-5 (set by
    1e-4 · min(r_s, r_p)), s_min^(-76) ≈ 10^344, which overflows float64
    well before the trapezoidal sum. Conversely, the prefactor r^(-2β) at
    those β values is huge while the integral is tiny, and their product
    is a finite physical answer. Doing everything in log-space sidesteps
    both extremes: the log-integrand is bounded, the cumulative integral
    is computed via reverse log-sum-exp, and the prefactor combines as a
    final subtraction in log before exponentiating.

    Speed: ~4× the linear-space version (~0.4 ms vs 0.1 ms at n_grid=4096),
    negligible against the outer projection cost.
    """
    r = np.atleast_1d(np.asarray(r, dtype=float))

    if s_inf is None:
        s_inf = 100.0 * max(r_s, r_p)

    s_min = 1e-4 * min(r_s, r_p)
    log_s = np.linspace(np.log(s_min), np.log(s_inf), n_grid)
    s = np.exp(log_s)

    # Log-space integrand. F(s) = ν(s) · G M(s) / s² is positive and smooth
    # everywhere on s > 0, so log F is finite on the whole grid.
    F = plummer_nu(s, r_p) * G_KPC_KMS2_MSUN * nfw_M(s, r_s, rho_s) / s ** 2
    # Trapezoid in log s integrates f(s) s d(log s); add the s factor in log:
    #   log[ s^(2β) · F(s) · s ] = 2β · log s + log F + log s
    log_g = 2.0 * beta * log_s + np.log(F) + log_s

    # Per-segment log-contribution to the trapezoid:
    #   log[ 0.5·(g[j] + g[j+1]) · dlog[j] ]
    # = logaddexp(log_g[j], log_g[j+1]) - log 2 + log dlog[j]
    dlog = np.diff(log_s)
    log_seg = np.logaddexp(log_g[:-1], log_g[1:]) - np.log(2.0) + np.log(dlog)

    # Reverse cumulative log-sum-exp:
    #   log_I[i] = log( sum_{j>=i} exp(log_seg[j]) )
    # = log( ∫_{s_i}^{s_inf} s^(2β) F(s) ds )
    log_I_seg = np.logaddexp.accumulate(log_seg[::-1])[::-1]   # length n_grid - 1
    # Endpoint: integral from s_inf to s_inf is zero ⇒ log = -inf.
    log_I = np.concatenate([log_I_seg, [-np.inf]])             # length n_grid

    # Interpolate log_I at log r. Clip r into [s_min, s_inf]; r < s_min is
    # outside our grid (caller should not be asking) and r > s_inf gives
    # log_I = -inf ⇒ result 0, which is the correct limit.
    log_r = np.log(np.clip(r, s_min, s_inf))
    log_I_at_r = np.interp(log_r, log_s, log_I)

    # Final assembly: ν σ_r²(r) = r^(-2β) · I(r) = exp( log_I - 2β · log r ).
    return np.exp(log_I_at_r - 2.0 * beta * log_r)


# ----------------------------------------------------------------------------
# Outer projection integral (P1): Σ(R) σ_los²(R)
# u-substitution to remove sqrt(r^2 - R^2) singularity; vectorized over R.
# ----------------------------------------------------------------------------

def Sigma_sigma_los2_quad(
    R: np.ndarray,
    beta: float,
    r_s: float,
    rho_s: float,
    r_p: float,
) -> np.ndarray:
    """
    Reference (slow, trusted) scipy.integrate.quad evaluation of Σ(R) σ_los²(R).

    Uses substitution r^2 = R^2 + u^2 to remove the sqrt singularity at r=R.
    The integrand in u is smooth on [0, ∞).

    For each R we still evaluate the inner ν σ_r² via scipy.quad (so this is
    very slow, but trustworthy). We use it only for cross-checking.
    """
    R = np.atleast_1d(np.asarray(R, dtype=float))
    out = np.empty_like(R)

    def integrand_u(u, Ri):
        r = np.sqrt(Ri ** 2 + u ** 2)
        nsr2 = nu_sigma_r2_quad(np.array([r]), beta, r_s, rho_s, r_p)[0]
        return (1.0 - beta * Ri ** 2 / r ** 2) * nsr2

    u_max = 100.0 * max(r_s, r_p)
    for i, Ri in enumerate(R):
        val, _ = integrate.quad(
            integrand_u, 0.0, u_max,
            args=(Ri,),
            limit=200,
            epsabs=0.0, epsrel=1e-7,
        )
        out[i] = 2.0 * val
    return out


def Sigma_sigma_los2_grid(
    R: np.ndarray,
    beta: float,
    r_s: float,
    rho_s: float,
    r_p: float,
    n_inner: int = 2048,
    n_outer: int = 512,
) -> np.ndarray:
    """
    Vectorized tabulated evaluation of Σ(R) σ_los²(R).

    Strategy:
      1. Tabulate ν σ_r²(r) on a shared log-spaced grid (one Jeans solve).
      2. For each R in the input, build a u-grid and trapezoidally integrate
         (1 - β R²/r²) ν σ_r²(r(u)) on that u-grid, with r = sqrt(R² + u²).

    The u-grid is log-spaced from u_min ~ 1e-4 * R to u_max = 100 * max(r_s, r_p),
    with an explicit u=0 endpoint included. Log spacing matters: the integrand
    is sharply peaked at small u (small u → r ≈ R, where ν is concentrated and
    the projection geometry weights mass directly along the line of sight),
    and a uniform u-grid undersamples this peak by orders of magnitude for
    R << r_p.

    Step 2 is vectorized across R via 2D broadcasting (n_R, n_outer).
    """
    R = np.atleast_1d(np.asarray(R, dtype=float))

    # Step 1: tabulate ν σ_r² on a log-r grid
    s_inf = 100.0 * max(r_s, r_p)
    s_min = 1e-4 * min(r_s, r_p)
    if np.any(R < s_min):
        raise ValueError(
            f"R contains values below the tabulation floor s_min={s_min:.3e} kpc "
            f"(= 1e-4 * min(r_s={r_s:.3e}, r_p={r_p:.3e})). "
            "Floor R before calling or increase n_inner."
        )
    log_r_tab = np.linspace(np.log(s_min), np.log(s_inf), n_inner)
    r_tab = np.exp(log_r_tab)
    nsr2_tab = nu_sigma_r2_grid(r_tab, beta, r_s, rho_s, r_p, n_grid=n_inner, s_inf=s_inf)

    # Step 2: log-spaced u-grid per R, fully vectorized over R.
    # Per-R u-grid: u in [0, u_max], logarithmically spaced for u > 0
    # plus an explicit 0 at the start. u_min depends on R (=1e-4·R) so the
    # u-grid is a 2D (n_R, n_outer) array.
    u_max = 100.0 * max(r_s, r_p)
    log_u_min = np.log(1e-4 * R)[:, None]                          # (n_R, 1)
    frac = np.linspace(0.0, 1.0, n_outer - 1)[None, :]              # (1, n_outer-1)
    log_u = log_u_min + frac * (np.log(u_max) - log_u_min)          # (n_R, n_outer-1)
    u = np.concatenate([np.zeros((R.size, 1)), np.exp(log_u)], axis=1)  # (n_R, n_outer)
    r_of_u = np.sqrt(R[:, None] ** 2 + u ** 2)                      # (n_R, n_outer)
    log_r_of_u = np.log(np.clip(r_of_u, s_min, s_inf))
    nsr2_at = np.interp(log_r_of_u.ravel(), log_r_tab, nsr2_tab).reshape(r_of_u.shape)
    integrand = (1.0 - beta * R[:, None] ** 2 / r_of_u ** 2) * nsr2_at
    return 2.0 * np.trapezoid(integrand, u, axis=1)


# ----------------------------------------------------------------------------
# Paired projection integral for Jeffreys-prior Fisher shape factor T.
# ----------------------------------------------------------------------------

def Sigma_sigma_los2_grid_pair(
    R: np.ndarray,
    beta: float,
    r_s: float,
    rho_s: float,
    r_p: float,
    n_inner: int = 2048,
    n_outer: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Tabulated evaluation of the projection integrals 𝒫(R) and 𝒬(R) for the
    Jeffreys-prior shape factor T = 3 − 𝒬/𝒫.

    Returns (P_arr, Q_arr), shape (N_R,), where:
        P_arr[i] = Σ(R_i) · σ_los²(R_i)                     [as in Sigma_sigma_los2_grid]
        Q_arr[i] = Σ(R_i) · σ_los_aux²(R_i)                  [same form, g(x) → h(x)]
    so that
        T_i = 3 − Q_arr[i] / P_arr[i].
    The 8πGρ_s r_s³ prefactor and Σ(R) cancel in Q/P, so they are not needed.

    Implementation reuses one shared inner log-`s` grid for both integrands
    (only the log[g(x)] vs log[h(x)] term differs) and one shared per-R
    log-`u` grid for the projection. The inner grid is run twice through
    reverse log-cumsum; the outer u-loop performs two interpolations and
    two trapezoids per R.
    """
    R = np.atleast_1d(np.asarray(R, dtype=float))

    s_inf = 100.0 * max(r_s, r_p)
    s_min = 1e-4 * min(r_s, r_p)
    if np.any(R < s_min):
        raise ValueError(
            f"R contains values below the tabulation floor s_min={s_min:.3e} kpc "
            f"(= 1e-4 * min(r_s={r_s:.3e}, r_p={r_p:.3e})). "
            "Floor R before calling or increase n_inner."
        )

    # Shared inner log-s grid.
    log_s = np.linspace(np.log(s_min), np.log(s_inf), n_inner)
    s = np.exp(log_s)
    x = s / r_s

    # Common log-integrand:  (2β − 1)·log s + log ν + log(G · 4π ρ_s r_s³).
    # (Derivation: trapezoid in log s integrates f·s·d log s; combined with the
    #  s^(2β) prefactor and the 1/s² in F = ν·G·M/s² the s-powers reduce to
    #  s^(2β−1).)
    log_pref = np.log(G_KPC_KMS2_MSUN * 4.0 * np.pi * rho_s * r_s ** 3)
    log_nu = np.log(plummer_nu(s, r_p))
    common = (2.0 * beta - 1.0) * log_s + log_nu + log_pref

    log_g_x = np.log(nfw_g(x))
    log_h_x = np.log(nfw_h(x))
    log_int_g = common + log_g_x
    log_int_h = common + log_h_x

    dlog = np.diff(log_s)
    log_dlog = np.log(dlog)
    half_log2 = np.log(2.0)

    log_seg_g = np.logaddexp(log_int_g[:-1], log_int_g[1:]) - half_log2 + log_dlog
    log_seg_h = np.logaddexp(log_int_h[:-1], log_int_h[1:]) - half_log2 + log_dlog

    log_I_g = np.concatenate([
        np.logaddexp.accumulate(log_seg_g[::-1])[::-1], [-np.inf],
    ])
    log_I_h = np.concatenate([
        np.logaddexp.accumulate(log_seg_h[::-1])[::-1], [-np.inf],
    ])

    # Tabulated ν σ_r² for both integrands at r = s_grid.
    nsr2_g_tab = np.exp(log_I_g - 2.0 * beta * log_s)
    nsr2_h_tab = np.exp(log_I_h - 2.0 * beta * log_s)

    # Shared per-R u-grid, fully vectorized over R; two interps + two trapezoids.
    log_r_tab = log_s
    u_max = 100.0 * max(r_s, r_p)
    log_u_min = np.log(1e-4 * R)[:, None]                          # (n_R, 1)
    frac = np.linspace(0.0, 1.0, n_outer - 1)[None, :]              # (1, n_outer-1)
    log_u = log_u_min + frac * (np.log(u_max) - log_u_min)          # (n_R, n_outer-1)
    u = np.concatenate([np.zeros((R.size, 1)), np.exp(log_u)], axis=1)
    r_of_u = np.sqrt(R[:, None] ** 2 + u ** 2)
    log_r_of_u = np.log(np.clip(r_of_u, s_min, s_inf))
    flat = log_r_of_u.ravel()
    nsr2_g_at = np.interp(flat, log_r_tab, nsr2_g_tab).reshape(r_of_u.shape)
    nsr2_h_at = np.interp(flat, log_r_tab, nsr2_h_tab).reshape(r_of_u.shape)
    kernel = 1.0 - beta * R[:, None] ** 2 / r_of_u ** 2
    P_arr = 2.0 * np.trapezoid(kernel * nsr2_g_at, u, axis=1)
    Q_arr = 2.0 * np.trapezoid(kernel * nsr2_h_at, u, axis=1)
    return P_arr, Q_arr


# ----------------------------------------------------------------------------
# Top-level: σ_los(R) at requested star radii.
# ----------------------------------------------------------------------------

def sigma_los(
    R: np.ndarray,
    beta: float,
    r_s: float,
    rho_s: float,
    r_p: float,
    method: str = "grid",
) -> np.ndarray:
    """
    Returns σ_los(R) [km/s] at each projected radius R [kpc].

    method='grid'  - vectorized tabulated trapezoidal (production)
    method='quad'  - scipy.integrate.quad reference (slow, for tests)
    """
    R = np.atleast_1d(np.asarray(R, dtype=float))
    if method == "grid":
        Ssl2 = Sigma_sigma_los2_grid(R, beta, r_s, rho_s, r_p)
    elif method == "quad":
        Ssl2 = Sigma_sigma_los2_quad(R, beta, r_s, rho_s, r_p)
    else:
        raise ValueError(f"unknown method {method!r}")
    Sigma = plummer_Sigma(R, r_p)
    return np.sqrt(Ssl2 / Sigma)


def sigma_los_with_T(
    R: np.ndarray,
    beta: float,
    r_s: float,
    rho_s: float,
    r_p: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (σ_los(R), T(R)) where T_i = 3 − 𝒬_i/𝒫_i is the Fisher-information
    shape factor entering the Jeffreys prior on (ln ρ_s, ln r_s).

    Single-pass evaluation: shares the inner log-s grid and the per-R u-grid
    between σ_los² and the auxiliary integral. ~2× the cost of `sigma_los`.
    """
    R = np.atleast_1d(np.asarray(R, dtype=float))
    P_arr, Q_arr = Sigma_sigma_los2_grid_pair(R, beta, r_s, rho_s, r_p)
    Sigma = plummer_Sigma(R, r_p)
    sigma_los_arr = np.sqrt(P_arr / Sigma)
    T_arr = 3.0 - Q_arr / P_arr
    return sigma_los_arr, T_arr
