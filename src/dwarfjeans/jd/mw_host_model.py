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

    MW2022_HOST.M_enc(D)     -> M_host(<D)          [Msun]   (mass within radius D)
    MW2022_HOST.dlnM_dlnr(D) -> gamma(D) = dln M_host / dln r   (dimensionless)

``MW2022_HOST`` is a ready-to-use default instance (the *full* MW2022 host, not
halo-only). Only numpy + scipy are required at
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
  < 1e-5. The assembled M_enc(D) reproduces an independent integration of gala's
  own total density over spheres to < 0.05% (max 4e-4 over 8--262 kpc). This
  parity is regression-guarded by tests/unit/test_mw_host_gala_parity.py against
  a committed fixture; regenerate with scripts/build_gala_reference.py (needs
  gala installed). gala is NOT a runtime dependency of the pipeline.
"""

from __future__ import annotations

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import CubicSpline

__all__ = ["MWPotential2022Host", "MW2022_HOST"]


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


# Ready-to-use default instance: the full MW2022 host. Consumed by
# dwarfjeans.jd.tidal as the default tidal-radius host.
MW2022_HOST = MWPotential2022Host()


if __name__ == "__main__":
    print(MW2022_HOST)
    print(f"{'r [kpc]':>10} {'M(<r) [Msun]':>16} {'gamma':>8} {'v_c [km/s]':>11}")
    for D in [1.0, 5.0, 8.122, 15.0, 20.0, 50.0, 100.0, 150.0, 200.0, 262.0]:
        print(f"{D:10.3f} {MW2022_HOST.M_enc(D):16.5e} "
              f"{MW2022_HOST.dlnM_dlnr(D):8.3f} {MW2022_HOST.v_circ(D):11.1f}")
