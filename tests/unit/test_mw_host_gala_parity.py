"""Parity: the numpy-only MW host reproduces gala's MilkyWayPotential2022.

Guards the "<0.2% vs gala 1.11" claim in mw_host_model.py / pipeline.tex against
a committed reference of the true spherically-enclosed mass, computed by
integrating gala's own density over spheres (scripts/build_gala_reference.py).
Requires no gala at test time -- it reads the fixture. A mis-transcribed
component constant would blow this far past 0.2%.
"""
from pathlib import Path

import numpy as np

from dwarfjeans.jd.mw_host_model import MW2022_HOST

FIXTURE = Path(__file__).parent / "data" / "gala_mw2022_Menc_reference.npz"


def test_M_enc_matches_gala_reference():
    z = np.load(FIXTURE)
    r = z["r_kpc"]
    M_ref = z["M_enc_Msun"]
    rel = np.abs(MW2022_HOST.M_enc(r) / M_ref - 1.0)
    assert rel.max() < 2.0e-3, (
        f"MW2022 host deviates from gala {str(z['gala_version'])} "
        f"spherically-enclosed mass by up to {rel.max():.2e} (> 0.2%): "
        f"per-radius {dict(zip(r.tolist(), rel.round(6).tolist()))}"
    )
