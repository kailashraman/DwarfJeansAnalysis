"""Unit tests for the SHMR-weighted per-dwarf SatGen prior."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from dwarfjeans.jeans.priors import (
    LOG10_RHOS_BOUNDS,
    LOG10_RS_BOUNDS,
    SATGEN_SHMR_DIR,
    V_HALFWIDTH,
    _load_satgen_table,
    _shmr_table_path,
    get_prior,
    make_satgen_shmr_prior_transform,
    make_satgen_shmr_prior_transform_with_nuisances,
)

REPO = Path(__file__).resolve().parents[2]

SHMRS = ("fattahi18", "moster18", "danieli23_const", "kim24")


def _table_for(shmr: str) -> Path:
    return SATGEN_SHMR_DIR / shmr / "segue_1.npz"


def _require_table(shmr: str) -> Path:
    path = _table_for(shmr)
    if not path.exists():
        pytest.skip(
            f"SHMR table {path} not built; run "
            f"scripts/build_satgen_shmr_prior_tables.py --shmr {shmr}"
        )
    return path


def _load_builder_module():
    """Load scripts/build_satgen_shmr_prior_tables.py as a module so the
    weighted helpers can be unit-tested without the h5 dependency."""
    name = "_build_satgen_shmr"
    if name in sys.modules:
        return sys.modules[name]
    path = REPO / "scripts" / "build_satgen_shmr_prior_tables.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------- registry & basic plumbing ----------

@pytest.mark.parametrize("shmr", SHMRS)
def test_registry_includes_satgen_shmr(shmr):
    p = get_prior("satgen_shmr", shmr=shmr, lvdb_key="segue_1")
    assert p.name == "satgen_shmr"
    assert p.needs_T is False


def test_get_prior_requires_kwargs():
    with pytest.raises(ValueError, match="requires kwargs"):
        get_prior("satgen_shmr")
    with pytest.raises(ValueError, match="requires kwargs"):
        get_prior("satgen_shmr", shmr="fattahi18")
    with pytest.raises(ValueError, match="unexpected kwargs"):
        get_prior("jeffreys", shmr="fattahi18", lvdb_key="segue_1")


@pytest.mark.parametrize("shmr", SHMRS)
def test_shmr_table_path(shmr):
    p = _shmr_table_path(shmr, "segue_1")
    assert p == _table_for(shmr)


# ---------- weighted statistics in the builder ----------

def test_weighted_mean_std_unweighted_matches_numpy():
    """With uniform weights, the reliability-Bessel correction must reduce
    to np.std(ddof=1)."""
    mod = _load_builder_module()
    rng = np.random.default_rng(0)
    y = rng.normal(loc=3.0, scale=2.0, size=500)
    w = np.ones_like(y)
    mu, sigma = mod._weighted_mean_std(y, w)
    assert mu == pytest.approx(np.mean(y))
    assert sigma == pytest.approx(np.std(y, ddof=1))


def test_weighted_mean_std_two_points_known_answer():
    """Closed form: with weights (w1, w2) on points (y1, y2),
    μ = (w1·y1 + w2·y2)/(w1+w2).
    The reliability-Bessel denominator is (Σw − Σw²/Σw),
    which for two points with weights w1, w2 simplifies to 2·w1·w2/(w1+w2)."""
    mod = _load_builder_module()
    y = np.array([1.0, 4.0])
    w = np.array([3.0, 1.0])
    sumw = w.sum()
    mu_exp = (w * y).sum() / sumw
    denom = sumw - (w ** 2).sum() / sumw  # = 2·w1·w2/(w1+w2) = 1.5 here
    var_exp = float((w * (y - mu_exp) ** 2).sum() / denom)
    mu, sigma = mod._weighted_mean_std(y, w)
    assert mu == pytest.approx(mu_exp)
    assert sigma == pytest.approx(np.sqrt(var_exp))


def test_weighted_quantile_bin_edges_uniform_weights():
    """With uniform weights, weighted quantile-cut edges land at the
    unweighted np.quantile values (up to interior midpoint convention)."""
    mod = _load_builder_module()
    rng = np.random.default_rng(0)
    x = np.sort(rng.uniform(-2.0, 1.0, size=10_000))
    w = np.ones_like(x)
    edges = mod._weighted_quantile_bin_edges(x, w, n_bins=20)
    # Endpoints are nudged outside the data; interior edges should bracket
    # the unweighted quantiles closely (within the inter-sample spacing).
    qs = np.linspace(0.0, 1.0, 21)
    ref = np.quantile(x, qs)
    interior = slice(1, -1)
    assert np.allclose(edges[interior], ref[interior], atol=5e-3)


def test_weighted_quantile_bin_edges_concentrate_weight():
    """Place a huge weight on a single x value; that x should land near
    the median (cumulative weight crosses 0.5 there)."""
    mod = _load_builder_module()
    x = np.linspace(0.0, 1.0, 1001)
    w = np.ones_like(x)
    w[500] = 1e6  # dominate the cumulative weight at x=0.5
    edges = mod._weighted_quantile_bin_edges(x, w, n_bins=10)
    # The 5th edge (quantile 0.5) must sit within +/- one sample of x[500].
    assert abs(edges[5] - 0.5) < 5e-3


# ---------- loader + transform ----------

@pytest.mark.parametrize("shmr", SHMRS)
def test_table_loader_finite_and_monotone(shmr):
    table = _require_table(shmr)
    grid, cdf, centers, mu, sigma = _load_satgen_table(str(table))
    assert grid.ndim == cdf.ndim == 1
    assert np.all(np.isfinite(cdf))
    assert np.all(np.diff(cdf) >= -1e-12)
    assert cdf[0] == pytest.approx(0.0, abs=1e-12)
    assert cdf[-1] == pytest.approx(1.0, abs=1e-12)
    assert np.all(np.isfinite(mu))
    assert np.all(np.isfinite(sigma))
    assert np.all(sigma > 0.0)


@pytest.mark.parametrize("shmr", SHMRS)
def test_transform_midpoint_in_bounds(shmr):
    _require_table(shmr)
    pt = make_satgen_shmr_prior_transform(
        V_center=210.0, shmr=shmr, lvdb_key="segue_1",
    )
    x = pt(np.array([0.5, 0.5, 0.5, 0.5]))
    assert np.isfinite(x).all()
    assert 210.0 - V_HALFWIDTH <= x[0] <= 210.0 + V_HALFWIDTH
    assert LOG10_RS_BOUNDS[0] <= x[1] <= LOG10_RS_BOUNDS[1]
    assert LOG10_RHOS_BOUNDS[0] <= x[2] <= LOG10_RHOS_BOUNDS[1]


@pytest.mark.parametrize("shmr", SHMRS)
def test_transform_samples_finite_across_cube(shmr):
    _require_table(shmr)
    pt = make_satgen_shmr_prior_transform(
        V_center=210.0, shmr=shmr, lvdb_key="segue_1",
    )
    rng = np.random.default_rng(0)
    U = rng.random((2000, 4))
    X = np.array([pt(u) for u in U])
    assert np.isfinite(X).all()
    assert X[:, 1].min() >= LOG10_RS_BOUNDS[0] - 1e-9
    assert X[:, 1].max() <= LOG10_RS_BOUNDS[1] + 1e-9


@pytest.mark.parametrize("shmr", SHMRS)
def test_transform_inverse_cdf_monotone_in_u1(shmr):
    """Holding u[2]=u[0]=u[3]=0.5, log10 r_s must be non-decreasing in u[1]."""
    _require_table(shmr)
    pt = make_satgen_shmr_prior_transform(
        V_center=210.0, shmr=shmr, lvdb_key="segue_1",
    )
    u1_grid = np.linspace(1e-6, 1 - 1e-6, 200)
    log10_rs = np.array([pt(np.array([0.5, u, 0.5, 0.5]))[1] for u in u1_grid])
    assert np.all(np.diff(log10_rs) >= -1e-9)


@pytest.mark.parametrize("shmr", SHMRS)
def test_nuisance_transform_runs(shmr):
    _require_table(shmr)
    pt = make_satgen_shmr_prior_transform_with_nuisances(
        V_center=210.0,
        d_mean=23.0, d_sigma=2.0,
        eps_mean=0.5, eps_sigma=0.1,
        rhalf_mean=4.0, rhalf_sigma=0.5,
        shmr=shmr, lvdb_key="segue_1",
    )
    x = pt(np.full(7, 0.5))
    assert x.shape == (7,)
    assert np.isfinite(x).all()


def test_missing_table_raises():
    """Loader must error if a table for the requested dwarf is missing."""
    with pytest.raises(FileNotFoundError):
        make_satgen_shmr_prior_transform(
            V_center=0.0, shmr="fattahi18", lvdb_key="not_a_dwarf",
        )
