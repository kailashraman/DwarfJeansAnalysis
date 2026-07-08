"""Tidal radius for dwarf-galaxy NFW halos in the SatGen m12 host.

`r_t` solves the Tormen, Diaferio & Syer (1998) / Springel et al. (2008) radial
tidal-radius condition (the convention named in docs/plan/stage3.md):

    r_t = D [ M_sub(<r_t) / ((factor - gamma(D)) * M_host(<D)) ]^(1/3),

with gamma(D) = dln M_host/dln r at the satellite's galactocentric distance D.
``factor`` is 2 for the radial tidal field (Tormen/Springel; default) and 3 for
the circular-orbit Jacobi/Hill convention (-> D (m/3M)^(1/3) for a point-mass
host). It is a swappable knob.

The equation is implicit (M_sub(<r_t) appears on the right), so it is solved
self-consistently per posterior draw with a 1-D root find.

The subhalo is the NFW (r_s, rho_s) of the Jeans posterior:
M_sub(<r) = 4 pi rho_s r_s^3 g(r/r_s). The default host is the real Milky Way
mass model ``MWPotential2022Host`` (Gala's ``MilkyWayPotential2022``: NFW halo +
Hernquist bulge + Hernquist nucleus + MN3 exponential disk; see
``mw_host_model.py``), which supplies M_host(<D) as the true spherically-enclosed
mass and gamma(D) analytically. The legacy fixed-mass NFW host ``HostNFW`` (SatGen
``m12res8-Diemer``: Mvir = 1e12 Msun, Rvir = 258.9 kpc, c ~ 11.5) is retained for
reproducing older chains; for that NFW, dln M/dln r = h(x)/g(x) with x = r/r_s,
reusing ``jeans.nfw_g`` / ``jeans.nfw_h``. Any host exposing ``M_enc(D)`` and
``dlnM_dlnr(D)`` works -- the solver is host-agnostic.

r_t depends only on (r_s, rho_s, D, host) -- not on the kinematics -- so it is
computed for every dwarf, resolved-sigma_los or not.
"""

from __future__ import annotations

from dataclasses import dataclass

import astropy.units as u
import numpy as np
from astropy.coordinates import Galactocentric, SkyCoord
from scipy.optimize import brentq

from dwarfjeans.jd import mw_host_model
from dwarfjeans.jd.mw_host_model import MW2022_HOST, MWPotential2022Host
from dwarfjeans.jeans import solver as jeans


@dataclass(frozen=True)
class HostNFW:
    """NFW host halo defined by virial mass, virial radius, and concentration.

    Defaults are the SatGen ``m12res8-Diemer`` host: a fixed-mass MW halo with
    Mvir = 1e12 Msun, Rvir = 258.9 kpc, and median concentration c ~ 11.5.
    """

    Mvir_Msun: float = 1.0e12
    Rvir_kpc: float = 258.9
    concentration: float = 11.5

    @property
    def r_s_kpc(self) -> float:
        return self.Rvir_kpc / self.concentration

    @property
    def rho_s_Msun_kpc3(self) -> float:
        # rho_s normalized so that M_host(<Rvir) = Mvir.
        g_c = float(jeans.nfw_g(self.concentration))
        return self.Mvir_Msun / (4.0 * np.pi * self.r_s_kpc ** 3 * g_c)

    def M_enc(self, r_kpc):
        """Host enclosed mass M_host(<r) [Msun]."""
        return jeans.nfw_M(r_kpc, self.r_s_kpc, self.rho_s_Msun_kpc3)

    def dlnM_dlnr(self, r_kpc):
        """Host enclosed-mass log-slope dln M_host/dln r at radius r."""
        x = np.asarray(r_kpc, float) / self.r_s_kpc
        return jeans.nfw_h(x) / jeans.nfw_g(x)


# Legacy fixed-mass NFW host: SatGen m12 (matches the satgen prior catalog).
# Retained to reproduce chains run before the real-MW host was adopted. See
# reference_satgen_m12_host memory / docs/plan/stage3.md.
SATGEN_M12_HOST = HostNFW()

# Default host for all new runs: the real Milky Way mass model (Gala
# MilkyWayPotential2022). See docs/plan/mw_host_model.md.
DEFAULT_HOST = MW2022_HOST

# Host tags recorded in chain metadata so a saved chain deserialises to the exact
# host it was run against (see host_save_fields / host_from_npz).
HOST_MODEL_MW2022 = "MWPotential2022"
HOST_MODEL_SATGEN_M12 = "SatGenM12NFW"


def host_save_fields(host) -> dict:
    """npz fields identifying ``host`` for chain persistence.

    The MW2022 host is a single fixed fiducial, so only a tag is stored; the
    legacy NFW host also stores its three scalars so it round-trips exactly.
    """
    if isinstance(host, MWPotential2022Host):
        return {"host_model": HOST_MODEL_MW2022}
    if isinstance(host, HostNFW):
        return {
            "host_model": HOST_MODEL_SATGEN_M12,
            "host_Mvir_Msun": float(host.Mvir_Msun),
            "host_Rvir_kpc": float(host.Rvir_kpc),
            "host_concentration": float(host.concentration),
        }
    raise TypeError(f"cannot serialise unknown host type {type(host)!r}")


def host_from_npz(z):
    """Reconstruct the host from a loaded chain npz, or None if absent.

    Dispatches on the ``host_model`` tag written by ``host_save_fields``. Legacy
    chains predate the tag and carry only ``host_Mvir_Msun`` etc.; those are the
    SatGen m12 NFW host by construction.
    """
    files = set(z.files)
    if "host_model" in files:
        model = str(z["host_model"])
        if model == HOST_MODEL_MW2022:
            return MW2022_HOST
        if model == HOST_MODEL_SATGEN_M12:
            return HostNFW(
                Mvir_Msun=float(z["host_Mvir_Msun"]),
                Rvir_kpc=float(z["host_Rvir_kpc"]),
                concentration=float(z["host_concentration"]),
            )
        raise ValueError(f"unknown host_model tag {model!r}")
    if "host_Mvir_Msun" in files:  # legacy chain, pre-tag: always the m12 NFW host
        return HostNFW(
            Mvir_Msun=float(z["host_Mvir_Msun"]),
            Rvir_kpc=float(z["host_Rvir_kpc"]),
            concentration=float(z["host_concentration"]),
        )
    return None


def galactocentric_distance(ra_deg: float, dec_deg: float, d_helio_kpc):
    """Galactocentric distance |r| [kpc] from heliocentric (ra, dec, distance).

    ``ra_deg``, ``dec_deg`` are scalars; ``d_helio_kpc`` may be a scalar or an
    array (e.g. the per-draw distance chain). Uses astropy's default
    Galactocentric frame (galcen_distance = 8.122 kpc, z_sun = 20.8 pc), which
    matches the convention SatGen_Dwarf uses for its galactocentric positions.
    """
    d = np.atleast_1d(np.asarray(d_helio_kpc, dtype=float))
    c = SkyCoord(ra=np.full(d.shape, ra_deg) * u.deg,
                 dec=np.full(d.shape, dec_deg) * u.deg,
                 distance=d * u.kpc)
    gc = c.transform_to(Galactocentric())
    r = np.sqrt(gc.x.to_value(u.kpc) ** 2
                + gc.y.to_value(u.kpc) ** 2
                + gc.z.to_value(u.kpc) ** 2)
    return float(r[0]) if np.isscalar(d_helio_kpc) or np.ndim(d_helio_kpc) == 0 else r


def tidal_radius(r_s_kpc: float, rho_s_Msun_kpc3: float, D_kpc: float,
                 host=DEFAULT_HOST, factor: float = 2.0) -> float:
    """Tormen/Springel tidal radius r_t [kpc] for one NFW subhalo.

    Solves r_t^3 * (factor - gamma(D)) * M_host(<D) = D^3 * M_sub(<r_t) for the
    non-trivial positive root, on log r_t with ``scipy.optimize.brentq``.

    ``factor`` = 2 (radial tidal field; default) or 3 (Jacobi/Hill). Returns NaN
    if the denominator is non-positive or no positive root brackets.
    """
    denom = (factor - float(host.dlnM_dlnr(D_kpc))) * float(host.M_enc(D_kpc))
    if not np.isfinite(denom) or denom <= 0.0 or not (D_kpc > 0.0):
        return float("nan")
    D3 = D_kpc ** 3

    def f(log_rt: float) -> float:
        rt = np.exp(log_rt)
        return rt ** 3 * denom - D3 * float(jeans.nfw_M(rt, r_s_kpc, rho_s_Msun_kpc3))

    # f(rt) < 0 for small rt (M_sub ~ rt^2 dominates rt^3) and > 0 for large rt
    # (rt^3 outgrows the ~ln rt NFW mass), so a unique positive root exists.
    lo, hi = np.log(1e-5 * r_s_kpc), np.log(1e4)
    f_lo, f_hi = f(lo), f(hi)
    if not (np.isfinite(f_lo) and np.isfinite(f_hi)) or f_lo * f_hi > 0.0:
        return float("nan")
    return float(np.exp(brentq(f, lo, hi, xtol=1e-6, rtol=1e-8)))


def tidal_radius_chain(r_s_kpc, rho_s_Msun_kpc3, D_kpc,
                       host=DEFAULT_HOST, factor: float = 2.0):
    """Vectorized ``tidal_radius`` over equal-length (r_s, rho_s, D) arrays."""
    r_s = np.atleast_1d(np.asarray(r_s_kpc, float))
    rho_s = np.atleast_1d(np.asarray(rho_s_Msun_kpc3, float))
    D = np.atleast_1d(np.asarray(D_kpc, float))
    out = np.empty(r_s.shape[0])
    for i in range(r_s.shape[0]):
        out[i] = tidal_radius(float(r_s[i]), float(rho_s[i]), float(D[i]),
                              host=host, factor=factor)
    return out
