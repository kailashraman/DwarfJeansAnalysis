"""Stage-3 deriver (src/dwarfjeans/postprocess.py).

Covers the single-source-of-truth contract: derive() is a deterministic pure
function of (chain, ctx), it agrees with the underlying primitives, and the
chain npz round-trips its reproducibility metadata.
"""

import numpy as np
import pytest

from dwarfjeans import postprocess as pp
from dwarfjeans.jeans import solver as jeans
from dwarfjeans.jd import factors as jdf
from dwarfjeans.jd import tidal as jdtidal


def _synthetic_chain(n=200, seed=1):
    """A plausible 7-column equal-weight chain for a UFD-ish halo."""
    rng = np.random.default_rng(seed)
    V = rng.normal(50.0, 1.0, n)
    log10_rs = rng.normal(np.log10(0.3), 0.1, n)
    log10_rhos = rng.normal(np.log10(3e8), 0.15, n)
    beta_tilde = rng.uniform(-0.3, 0.3, n)
    d = rng.normal(30.0, 1.0, n)
    eps = np.zeros(n)
    rhalf_arcmin = rng.normal(5.7, 0.2, n)
    return np.column_stack([V, log10_rs, log10_rhos, beta_tilde, d, eps, rhalf_arcmin])


def _minimal_ctx(n_stars=40):
    """A GalaxyContext with only the fields derive() reads populated."""
    rng = np.random.default_rng(0)
    Rad_arcmin = np.sort(rng.uniform(0.5, 10.0, n_stars))
    row = {"ra_deg": 150.0, "dec_deg": -50.0}
    return pp.GalaxyContext(
        lvdb_key="test_galaxy", row=row, nuisance_priors={}, d_kpc=30.0,
        V_center=50.0, V_halfwidth=10.0, galaxy={},
        sigma_los_walker={"q16": 3.0, "median": 3.5, "q84": 4.0},
        perspective_kwargs={}, selection_audit={}, Rad_arcmin=Rad_arcmin,
    )


def test_derive_is_deterministic():
    chain = _synthetic_chain()
    ctx = _minimal_ctx()
    # thin_jd/thin_sigma < N so the RNG thinning is actually exercised.
    a = pp.derive(chain, ctx, rseed=7, thin_jd=50, thin_sigma=80, thin_profile=40)
    b = pp.derive(chain, ctx, rseed=7, thin_jd=50, thin_sigma=80, thin_profile=40)
    for key in ("r_t_chain", "theta95_J", "sigma_at_Rhalf",
                "R_GC_chain", "idx_jd", "idx_sig"):
        np.testing.assert_array_equal(a[key], b[key],
                                      err_msg=f"{key} not reproducible at fixed rseed")
    np.testing.assert_array_equal(a["log10_J"]["0p5deg"], b["log10_J"]["0p5deg"])


def test_derive_thinning_depends_on_seed():
    chain = _synthetic_chain()
    ctx = _minimal_ctx()
    a = pp.derive(chain, ctx, rseed=1, thin_jd=50)
    b = pp.derive(chain, ctx, rseed=2, thin_jd=50)
    # Different seeds -> different thinned subsets (thin_jd=50 < N=200).
    assert not np.array_equal(a["idx_jd"], b["idx_jd"])


def test_derive_matches_primitives():
    chain = _synthetic_chain()
    ctx = _minimal_ctx()
    d = pp.derive(chain, ctx, rseed=3, thin_jd=50)

    rs = 10.0 ** chain[:, 1]
    rhos = 10.0 ** chain[:, 2]
    # M(r_half) is a full-chain quantity: must match nfw_M exactly.
    np.testing.assert_allclose(
        d["log10_M_3d"],
        np.log10(jeans.nfw_M(d["r_half_3d_chain"], rs, rhos)), rtol=1e-12)

    # J(0.5deg) on the thinned set must match J_D_factors at the per-draw r_t.
    idx = d["idx_jd"]
    R_GC = d["R_GC_chain"]
    for k in (0, len(idx) // 2, len(idx) - 1):
        i = idx[k]
        r_t = jdtidal.tidal_radius(float(rs[i]), float(rhos[i]), float(R_GC[k]))
        J, _ = jdf.J_D_factors(0.5 * jdf.DEG, float(chain[i, 4]),
                               float(rs[i]), float(rhos[i]), r_t)
        expect = np.log10(J) + jdf.LOG10_J_FAC
        assert d["log10_J"]["0p5deg"][k] == pytest.approx(expect, rel=1e-10)
        assert d["r_t_chain"][k] == pytest.approx(r_t, rel=1e-10)


def test_save_load_chain_roundtrip(tmp_path):
    chain = _synthetic_chain()
    ctx = _minimal_ctx()
    path = tmp_path / "posterior_samples.npz"
    pp.save_chain(path, chain, ctx, rseed=11, prior_name="jeffreys", shmr=None,
                  thin_sigma=2000, thin_jd=500, thin_profile=300,
                  p_min=0.5, rmax_over_rhalf=2.0, drop_variable=True,
                  use_p_weights=False)
    samples, meta = pp.load_chain(path)
    np.testing.assert_array_equal(samples, chain)
    assert meta["rseed"] == 11
    assert meta["thin_jd"] == 500
    assert meta["lvdb_key"] == "test_galaxy"
    assert meta["selection"] == {
        "p_min": 0.5, "rmax_over_rhalf": 2.0,
        "drop_variable": True, "use_p_weights": False}
    assert meta["tidal_factor"] == 2.0
    assert isinstance(meta["host"], jdtidal.HostNFW)
    assert meta["host"].Mvir_Msun == jdtidal.SATGEN_M12_HOST.Mvir_Msun
    # Chain-only contract: no derived arrays persisted.
    z = np.load(path, allow_pickle=True)
    for forbidden in ("log10_J_0p5deg", "log10_M_half_3d", "r_t_kpc_chain",
                      "theta95_J_deg", "sigma_los_at_Rhalf2d"):
        assert forbidden not in z.files


def test_save_derived_export(tmp_path):
    """save_derived writes the per-draw derived arrays (the external-consumer
    export) with the legacy key names plus theta95_J_deg, with the documented
    length contract (full chain vs idx_jd-thinned)."""
    chain = _synthetic_chain()
    ctx = _minimal_ctx()
    d = pp.derive(chain, ctx, rseed=5, thin_jd=50, thin_sigma=80, thin_profile=40)
    path = tmp_path / "derived.npz"
    pp.save_derived(path, d, ctx)

    z = np.load(path, allow_pickle=True)
    # Keys Jdata.py reads + the new theta95.
    for required in ("log10_rs", "log10_rhos", "beta", "log10_J_0p5deg",
                     "theta95_J_deg", "r_t_kpc", "idx_jd"):
        assert required in z.files, f"{required} missing from derived.npz"

    n = chain.shape[0]
    n_jd = d["idx_jd"].size
    # Full-chain-length blocks.
    assert z["V"].shape == (n,)
    assert z["log10_rs"].shape == (n,)
    # idx_jd-thinned J/D block.
    assert z["log10_J_0p5deg"].shape == (n_jd,)
    assert z["theta95_J_deg"].shape == (n_jd,)
    assert z["r_t_kpc"].shape == (n_jd,)
    # theta95 is exported in degrees (radians / DEG).
    np.testing.assert_allclose(z["theta95_J_deg"], d["theta95_J"] / jdf.DEG)
    # No PM columns for a 7-column (PM-less) chain.
    assert "pmra" not in z.files


def test_resolve_run_meta_backfills_legacy_from_audit(tmp_path):
    """A legacy npz (no reproducibility metadata) + an audit.json must resolve
    to the AUDIT's prior/selection/rseed, never silent defaults. Guards the
    plot/reprocess drift where one view backfilled and the other defaulted."""
    import json
    run_dir = tmp_path / "leo_x" / "loguniform"
    run_dir.mkdir(parents=True)
    # Legacy-style npz: only the chain, no selection_*/rseed/prior_name keys.
    np.savez(run_dir / "posterior_samples.npz", samples_eq=_synthetic_chain())
    (run_dir / "audit.json").write_text(json.dumps({
        "lvdb_key": "leo_x",
        "prior_name": "loguniform",
        "selection_policy": {"p_min": 0.8, "R_over_rhalf_max": 1.5,
                             "drop_variable": False},
        "dynesty": {"rseed": 7},
        "thinning": {"sigma": 1000, "profile": 100, "jd": 250},
    }))

    _samples, meta = pp.resolve_run_meta(run_dir)
    # The non-default audit values must win over the function defaults.
    assert meta["prior_name"] == "loguniform"
    assert meta["lvdb_key"] == "leo_x"
    assert meta["rseed"] == 7
    assert meta["thin_jd"] == 250
    assert meta["selection"] == {
        "p_min": 0.8, "rmax_over_rhalf": 1.5,
        "drop_variable": False, "use_p_weights": False}


def test_resolve_run_meta_chain_only(tmp_path):
    """A new chain-only npz resolves straight from its own metadata."""
    chain = _synthetic_chain()
    ctx = _minimal_ctx()
    run_dir = tmp_path / "leo_x" / "jeffreys"
    run_dir.mkdir(parents=True)
    pp.save_chain(run_dir / "posterior_samples.npz", chain, ctx, rseed=3,
                  prior_name="jeffreys", shmr=None, thin_sigma=2000,
                  thin_jd=500, thin_profile=300, p_min=0.5, rmax_over_rhalf=2.0,
                  drop_variable=True, use_p_weights=False)
    _samples, meta = pp.resolve_run_meta(run_dir)
    assert meta["rseed"] == 3 and meta["lvdb_key"] == "test_galaxy"


def test_derive_rejects_chain_with_wrong_column_count():
    """ctx.has_pm disagreeing with the chain's columns must fail loudly."""
    chain7 = _synthetic_chain()              # 7 columns, ctx.has_pm False -> ok
    ctx = _minimal_ctx()
    pp.derive(chain7, ctx, rseed=0, thin_jd=20, thin_sigma=20, thin_profile=20)
    chain9 = np.column_stack([chain7, np.zeros(len(chain7)), np.zeros(len(chain7))])
    with pytest.raises(ValueError, match="disagrees with the saved chain"):
        pp.derive(chain9, ctx, rseed=0)


def test_summary_rows_include_new_quantities():
    chain = _synthetic_chain()
    ctx = _minimal_ctx()
    d = pp.derive(chain, ctx, rseed=0)
    names = {r[0] for r in pp.summary_rows(d, ctx)}
    for required in ("r_t_kpc", "theta95_J_deg", "R_GC_kpc",
                     "log10_J_0p5deg_GeV2_cm5", "sigma_los_walker_kms"):
        assert required in names
