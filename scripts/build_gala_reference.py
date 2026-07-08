"""Regenerate the gala parity fixture for ``dwarfjeans.jd.mw_host_model``.

Computes the true spherically-enclosed mass ``M(<r) = int rho dV`` of gala's
``MilkyWayPotential2022`` by integrating gala's OWN total density over spheres,
using a quadrature (vectorised Simpson over the polar angle + cumulative Simpson
over radius) that is fully independent of the leggauss shell-integral +
cubic-spline mass table in ``mw_host_model.py``. The resulting table is committed
as a fixture so ``test_mw_host_gala_parity`` can assert the numpy-only module
reproduces gala WITHOUT requiring gala at test time.

NOTE: this is a spherical ``int rho dV`` mass, NOT gala's force-based
``potential.mass_enclosed`` (~ r v_c^2 / G) -- the latter is a different quantity
for the flattened disk and is deliberately not what the tidal criterion uses.

Requires gala (``pip install gala`` into the DwarfJeans env). Run:
    python scripts/build_gala_reference.py
"""
from __future__ import annotations

import warnings
from pathlib import Path

import astropy.units as u
import numpy as np
from scipy.integrate import cumulative_simpson, simpson

warnings.filterwarnings("ignore")  # silence gala's MW2022 deprecation notice

import gala  # noqa: E402
from gala.potential import MilkyWayPotential2022  # noqa: E402

GALA_VERSION = gala.__version__          # label the fixture with the actual version
RHO = u.Msun / u.kpc ** 3
FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "unit" / "data" \
    / "gala_mw2022_Menc_reference.npz"

# Reporting radii: the doc's reference points plus the satellite band.
TARGETS = np.array([8.122, 15.0, 20.0, 30.0, 50.0, 80.0, 100.0, 150.0, 200.0, 262.0])


def main() -> None:
    pot = MilkyWayPotential2022()
    rg = np.linspace(1e-3, 262.0, 20000)     # M(<1e-3 kpc) is negligible (~1e5 Msun)
    # Put the reporting radii exactly on the grid so the final np.interp is exact
    # (off-grid interpolation would otherwise inflate the reported deviation at
    # small r, conflating interp noise with true model-vs-gala discrepancy).
    rg = np.unique(np.concatenate([rg, TARGETS]))
    th = np.linspace(0.0, np.pi, 401)        # odd count for Simpson; theta-grid
    sinth = np.sin(th)                       # convergence checked (401->6401 stable)
    costh = np.cos(th)

    shell = np.empty_like(rg)                # 2 pi * int_0^pi sin(th) rho dth
    for i, r in enumerate(rg):
        q = np.vstack([r * sinth, np.zeros_like(sinth), r * costh]) * u.kpc
        rho = pot.density(q).to_value(RHO)
        shell[i] = 2.0 * np.pi * simpson(sinth * rho, x=th)

    m_cum = cumulative_simpson(rg ** 2 * shell, x=rg, initial=0.0)
    m_ref = np.interp(TARGETS, rg, m_cum)

    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(FIXTURE, r_kpc=TARGETS, M_enc_Msun=m_ref,
             gala_version=GALA_VERSION, potential="MilkyWayPotential2022")

    from dwarfjeans.jd.mw_host_model import MW2022_HOST
    dev = np.abs(MW2022_HOST.M_enc(TARGETS) / m_ref - 1.0)
    print(f"wrote {FIXTURE}")
    print(f"gala {GALA_VERSION} MilkyWayPotential2022; max |module/gala - 1| = "
          f"{dev.max():.3e}")


if __name__ == "__main__":
    main()
