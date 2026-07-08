# Milky Way Host Mass Model (`mw_host_model.py`)

> **Implemented** at `src/dwarfjeans/jd/mw_host_model.py` and wired in as
> `tidal.DEFAULT_HOST` (the module-level instance is named `MW2022_HOST`, not
> `HostNFW`, to avoid colliding with the legacy NFW host class in `tidal.py`).

Self-contained implementation of Gala's **`MilkyWayPotential2022`**, in full:
NFW halo + Hernquist bulge + Hernquist nucleus + 3-Miyamoto–Nagai exponential disk.
Depends only on **numpy + scipy** (astropy optional). No `gala` import at runtime.

Provides exactly two quantities as functions of galactocentric radius `D` (kpc):

- `HostNFW.M_enc(D)` → `M_host(<D)` in M_sun (mass within a sphere of radius D)
- `HostNFW.dlnM_dlnr(D)` → `gamma(D) = dln M / dln r` (dimensionless)

**Verified against gala 1.11:** the hand-coded MN3 disk density reproduces
`disk.density()` to <1e-6; the NFW mass matches `mass_enclosed` to <1e-5; the
assembled `M_enc(D)` matches a brute-force integration of gala's own total
density over spheres to ~0.2%.

## How to use

```python
# deps: pip install numpy scipy   (astropy optional)
from mw_host_model import HostNFW      # ready-to-use instance (full MW2022 host); `Host` is an alias

D = 55.0                       # kpc: float, array-like, or astropy Quantity
M = HostNFW.M_enc(D)           # -> M_host(<D) in Msun
g = HostNFW.dlnM_dlnr(D)       # -> gamma(D) = dln M / dln r

import numpy as np             # vectorized:
r = np.linspace(10, 300, 100)  # kpc
M = HostNFW.M_enc(r)           # ndarray, Msun
g = HostNFW.dlnM_dlnr(r)       # ndarray
```

### Contract / behavior

- Input `D` in **kpc** (scalar, array-like, or astropy `Quantity`). Output `M_enc` in **M_sun**, `dlnM_dlnr` dimensionless. Scalar in → scalar out; array in → array out.
- `D <= 0` → `M_enc = 0`, `dlnM_dlnr = nan`. Valid to arbitrarily large `D` (halo analytic; disk held at its exact total beyond 500 kpc).
- `M_enc(D)` is the **true spherically-enclosed mass** ∫ρ dV inside radius D — *not* gala's force-based `mass_enclosed` (≈ r·v_c²/G). The two agree once the halo dominates (r ≳ 100 kpc) but differ ~3% at 8 kpc where the disk matters. Switch conventions if your formula needs the force/rotation-curve definition.
- `dlnM_dlnr` uses analytic derivatives for the three spherical components and the exact shell integral for the disk (no finite differencing).
- Extras: `HostNFW.v_circ(D)` (km/s, spherical-average diagnostic only), `HostNFW.rho(R, z)` (total density, M_sun/kpc³).

### Reference values (sanity check)

| r [kpc] | M(<r) [M_sun] | gamma | v_c [km/s] |
|--------:|--------------:|------:|-----------:|
| 8.122   | 8.78e10       | 0.99  | 216        |
| 50      | 4.32e11       | 0.75  | 193        |
| 100     | 6.89e11       | 0.60  | 172        |
| 200     | 1.00e12       | 0.48  | 147        |

## The module

Save the block below as `mw_host_model.py`.

```python
"""
mw_host_model.py
================

Self-contained Milky Way host mass model = Gala's ``MilkyWayPotential2022``,
in full and unmodified. No stitching of external components.

``MilkyWayPotential2022`` (Price-Whelan et al.; halo/disk fit to a compilation
of MW measurements) is a composite of four pieces:

    halo     : NFW              m = 5.5427e11 Msun, r_s = 15.626 kpc      (spherical)
    bulge    : Hernquist        m = 5.0000e09 Msun, c   = 1.0    kpc      (spherical)
    nucleus  : Hernquist        m = 1.8142e09 Msun, c   = 0.0688867 kpc   (spherical)
    disk     : MN3ExponentialDisk (sum of 3 Miyamoto-Nagai disks)          (flattened)
               m = 4.7717e10 Msun (target), h_R = 2.6 kpc, h_z = 0.3 kpc

This module exposes exactly the two quantities you asked for, as functions of a
(spherical) galactocentric distance ``D`` in kpc:

    HostNFW.M_enc(D)       -> M_host(<D)          [Msun]   (mass within radius D)
    HostNFW.dlnM_dlnr(D)   -> gamma(D) = dln M_host / dln r   (dimensionless)

``HostNFW`` is a ready-to-use default instance (name kept for API compatibility;
it is the *full* MW2022 host, not halo-only). Only numpy + scipy are required at
runtime -- gala is not imported. Every constant and formula below was extracted
from and verified against gala 1.11 (see "Provenance & verification").

Definitions
-----------
* ``M_enc(D)`` is the SPHERICALLY enclosed mass: all mass inside a sphere of
  radius D. The three spherical components (halo, bulge, nucleus) contribute in
  closed form; the flattened disk is integrated in 3D over the sphere.
* ``gamma(D) = d ln M(<r)/d ln r = r*(dM/dr)/M(<r)``, evaluated at r = D.

Provenance & verification
-------------------------
* Halo/bulge/nucleus parameters read directly from
  ``MilkyWayPotential2022()['halo'|'bulge'|'nucleus'].parameters``.
* The disk is Smith et al. (2015)'s 3-Miyamoto-Nagai representation. Its three
  (M_i, a_i) and shared b were read from gala's internal ``disk._ms/_as/_b``;
  the hand-coded MN3 density below reproduces ``disk.density()`` to < 1e-6 at all
  tested points. Note M_2 < 0 (a genuine MN3 feature) and the realized disk mass
  sum(M_i) = 5.2866e10 differs from the nominal target 4.7717e10.
* NFW convention: M(<r) = m[ln(1+x) - x/(1+x)] matches gala's mass_enclosed to
  < 1e-5. The assembled M_enc(D) reproduces a brute-force integration of gala's
  own total density over spheres to ~0.2% (limited by the brute-force grid, not
  this model).
"""

from __future__ import annotations

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import CubicSpline

__all__ = ["MWPotential2022Host", "HostNFW", "Host"]


# --------------------------------------------------------------------------
# MilkyWayPotential2022 component parameters (from gala 1.11)
# --------------------------------------------------------------------------
_HALO = dict(m=554270000000.0, r_s=15.626)                 # NFW  [Msun, kpc]
_BULGE = dict(m=5000000000.0, c=1.0)                       # Hernquist [Msun, kpc]
_NUCLEUS = dict(m=1814200000.0, c=0.0688867)               # Hernquist [Msun, kpc]

# Disk: MN3ExponentialDiskPotential -> three Miyamoto-Nagai disks (shared b).
# a_i [kpc], M_i [Msun] (M_i[1] is negative by construction), b [kpc].
_DISK_A = np.array([1.5259431976529216, 6.782764436261113, 5.894799616164217])
_DISK_M = np.array([7872306998.700792, -275625221944.33154, 320618418897.9487])
_DISK_B = 0.20663742603550295


def _to_kpc(D):
    """Accept float/array in kpc, or an astropy Quantity; return a plain kpc array."""
    try:
        import astropy.units as u
        if isinstance(D, u.Quantity):
            D = D.to_value(u.kpc)
    except Exception:
        pass
    return np.asarray(D, dtype=float)


class MWPotential2022Host:
    """
    Full MilkyWayPotential2022 as a spherically-enclosed-mass model.

    Parameters
    ----------
    r_min, r_max : float
        Radial range [kpc] over which the (flattened) disk enclosed mass is
        tabulated. The spherical components are analytic at all radii; beyond
        r_max the disk mass is held at its exact total sum(M_i).
    n_grid : int
        Number of log-spaced radial nodes for the disk mass table.
    n_theta : int
        Gauss-Legendre nodes for the polar (shell) integral of the disk.
    """

    def __init__(self, r_min=1e-4, r_max=500.0, n_grid=6000, n_theta=64):
        self.m_halo, self.r_s = _HALO["m"], _HALO["r_s"]
        self.m_bulge, self.c_bulge = _BULGE["m"], _BULGE["c"]
        self.m_nucleus, self.c_nucleus = _NUCLEUS["m"], _NUCLEUS["c"]
        self.disk_a, self.disk_M, self.disk_b = _DISK_A, _DISK_M, _DISK_B
        self.disk_mass_total = float(self.disk_M.sum())  # exact realized disk mass
        self._r_min, self._r_max = float(r_min), float(r_max)

        # Gauss-Legendre nodes for theta in [0, pi/2]; x2 covers the z<0 hemisphere.
        x, w = leggauss(int(n_theta))
        self._sin = np.sin(0.25 * np.pi * (x + 1.0))
        self._cos = np.cos(0.25 * np.pi * (x + 1.0))
        self._wt = 0.25 * np.pi * w

        self._build_disk_table(int(n_grid))

    # ---- component densities --------------------------------------------
    def rho_halo(self, r):
        r = np.asarray(r, float); x = r / self.r_s
        rho_s = self.m_halo / (4 * np.pi * self.r_s ** 3)
        return rho_s / (x * (1 + x) ** 2)

    @staticmethod
    def _rho_hernquist(r, m, c):
        r = np.asarray(r, float)
        return m * c / (2 * np.pi) / (r * (r + c) ** 3)

    def rho_disk(self, R, z):
        """MN3 disk density [Msun/kpc^3] (sum of 3 Miyamoto-Nagai disks)."""
        R = np.asarray(R, float); z = np.asarray(z, float)
        b = self.disk_b
        zb = np.sqrt(z * z + b * b)
        out = np.zeros(np.broadcast(R, z).shape)
        for M, a in zip(self.disk_M, self.disk_a):
            ap = a + zb
            out = out + M * b * b * (a * R * R + (a + 3 * zb) * ap * ap) \
                / (4 * np.pi * (R * R + ap * ap) ** 2.5 * zb ** 3)
        return out

    def rho(self, R, z):
        """Total MW2022 density [Msun/kpc^3] at cylindrical (R, z) [kpc]."""
        r = np.sqrt(np.asarray(R, float) ** 2 + np.asarray(z, float) ** 2)
        return (self.rho_halo(r)
                + self._rho_hernquist(r, self.m_bulge, self.c_bulge)
                + self._rho_hernquist(r, self.m_nucleus, self.c_nucleus)
                + self.rho_disk(R, z))

    # ---- analytic spherical enclosed masses & derivatives ----------------
    def _M_halo(self, r):
        x = r / self.r_s
        return self.m_halo * (np.log1p(x) - x / (1 + x))

    def _dM_halo(self, r):
        x = r / self.r_s
        return self.m_halo * r / (self.r_s ** 2 * (1 + x) ** 2)

    @staticmethod
    def _M_hern(r, m, c):
        return m * r ** 2 / (r + c) ** 2

    @staticmethod
    def _dM_hern(r, m, c):
        return 2 * m * r * c / (r + c) ** 3

    # ---- flattened disk: shell integral ---------------------------------
    def _shell_sigma_disk(self, r):
        """sigma(r) = int_0^pi sin(theta) rho_disk(r sin, r cos) dtheta."""
        r = np.atleast_1d(np.asarray(r, float))
        R = np.outer(r, self._sin)
        Z = np.outer(r, self._cos)
        return 2.0 * ((self.rho_disk(R, Z) * self._sin) * self._wt).sum(axis=1)

    def _dMdr_disk(self, r):
        r = np.atleast_1d(np.asarray(r, float))
        return 2 * np.pi * r ** 2 * self._shell_sigma_disk(r)

    def _build_disk_table(self, n_grid):
        rg = np.logspace(np.log10(self._r_min), np.log10(self._r_max), n_grid)
        integrand = self._dMdr_disk(rg)
        M0 = (2 * np.pi / 3.0) * self._shell_sigma_disk(np.array([self._r_min]))[0] * self._r_min ** 3
        Md = M0 + cumulative_trapezoid(integrand, rg, initial=0.0)
        self._rg = rg
        self._M0 = M0
        self._Md_spline = CubicSpline(rg, Md)

    def _M_disk(self, r):
        r = np.asarray(r, float)
        rmin, rmax = self._r_min, self._r_max
        inner = self._M0 * (r / rmin) ** 3
        mid = self._Md_spline(np.clip(r, rmin, rmax))
        # beyond the grid, use the exact realized total mass sum(M_i)
        return np.where(r < rmin, inner,
                        np.where(r >= rmax, self.disk_mass_total, mid))

    def _dMdr_disk_eval(self, r):
        r = np.asarray(r, float)
        rmin, rmax = self._r_min, self._r_max
        inner = 3 * self._M0 * (r / rmin) ** 3 / np.maximum(r, 1e-300)
        mid = self._dMdr_disk(np.clip(r, rmin, rmax)).reshape(r.shape) if r.ndim \
            else self._dMdr_disk(np.array([np.clip(r, rmin, rmax)]))[0]
        return np.where(r < rmin, inner, np.where(r >= rmax, 0.0, np.asarray(mid, float)))

    # ---- public API ------------------------------------------------------
    def M_enc(self, D):
        """
        Spherically enclosed host mass M_host(<D).

        Parameters
        ----------
        D : float, array-like, or astropy Quantity   (galactocentric radius, kpc)

        Returns
        -------
        float or ndarray : enclosed mass in solar masses (Msun).
        """
        D = _to_kpc(D)
        scalar = (D.ndim == 0)
        r = np.atleast_1d(D)
        M = (self._M_halo(r)
             + self._M_hern(r, self.m_bulge, self.c_bulge)
             + self._M_hern(r, self.m_nucleus, self.c_nucleus)
             + self._M_disk(r))
        M = np.where(r <= 0.0, 0.0, M)
        return float(M[0]) if scalar else M

    def dlnM_dlnr(self, D):
        """
        gamma(D) = dln M(<r)/dln r at r = D  (dimensionless).

        Parameters
        ----------
        D : float, array-like, or astropy Quantity   (galactocentric radius, kpc)
        """
        D = _to_kpc(D)
        scalar = (D.ndim == 0)
        r = np.atleast_1d(D).astype(float)
        dM = (self._dM_halo(r)
              + self._dM_hern(r, self.m_bulge, self.c_bulge)
              + self._dM_hern(r, self.m_nucleus, self.c_nucleus)
              + self._dMdr_disk_eval(r))
        M = (self._M_halo(r)
             + self._M_hern(r, self.m_bulge, self.c_bulge)
             + self._M_hern(r, self.m_nucleus, self.c_nucleus)
             + self._M_disk(r))
        with np.errstate(divide="ignore", invalid="ignore"):
            gamma = r * dM / M
        gamma = np.where(r <= 0.0, np.nan, gamma)
        return float(gamma[0]) if scalar else gamma

    # ---- convenience -----------------------------------------------------
    def v_circ(self, D):
        """Spherically-enclosed circular speed sqrt(G M(<D)/D) in km/s (D in kpc).

        Diagnostic only: this is the spherical-average proxy, not the exact
        in-plane rotation speed of the flattened disk.
        """
        G = 4.30091727e-6  # kpc (km/s)^2 / Msun
        D = _to_kpc(D)
        return np.sqrt(G * self.M_enc(D) / D)

    def __repr__(self):
        return ("MWPotential2022Host(NFW halo + Hernquist bulge + Hernquist "
                f"nucleus + MN3 disk; disk_mass_total={self.disk_mass_total:.4e} Msun)")


# Ready-to-use default instance. Name kept for API compatibility with existing
# code that calls HostNFW.M_enc(D) / HostNFW.dlnM_dlnr(D); it is the full MW2022 host.
HostNFW = MWPotential2022Host()
Host = HostNFW  # clearer alias


if __name__ == "__main__":
    print(HostNFW)
    print(f"{'r [kpc]':>10} {'M(<r) [Msun]':>16} {'gamma':>8} {'v_c [km/s]':>11}")
    for D in [1.0, 5.0, 8.122, 15.0, 20.0, 50.0, 100.0, 150.0, 200.0, 262.0]:
        print(f"{D:10.3f} {HostNFW.M_enc(D):16.5e} "
              f"{HostNFW.dlnM_dlnr(D):8.3f} {HostNFW.v_circ(D):11.1f}")
```
