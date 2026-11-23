"""run_inference must be reproducible given (rseed, npool), including the
equal-weight resample samples_eq, not just logZ."""

import numpy as np

from dwarfjeans.jeans import inference


def _toy_loglike(*args, **kwargs):
    def loglike(theta):
        return float(-0.5 * np.sum((np.asarray(theta) - 0.5) ** 2 / 0.05**2))
    return loglike


def _run(monkeypatch, rseed, npool=1):
    monkeypatch.setattr(inference, "make_loglike_asimov", _toy_loglike)
    galaxy = {"R": None, "sigma_eps": None, "p": None,
              "truth": {"r_p": 1.0}, "sigma_los_true": None}
    from dwarfjeans.jeans.priors import get_prior
    prior = get_prior("loguniform")
    monkeypatch.setattr(type(prior), "make_transform",
                        lambda self, *a, **k: (lambda u: np.asarray(u)),
                        raising=False)
    return inference.run_inference(
        galaxy, nlive=50, dlogz=1.0, rseed=rseed, asimov=True,
        prior_name="loguniform", npool=npool,
    )


def test_samples_eq_reproducible_given_rseed(monkeypatch):
    a = _run(monkeypatch, 3)
    b = _run(monkeypatch, 3)
    assert a["logz"] == b["logz"]
    assert np.array_equal(a["samples_eq"], b["samples_eq"])


def test_samples_eq_differs_between_rseeds(monkeypatch):
    a = _run(monkeypatch, 3)
    b = _run(monkeypatch, 4)
    assert not np.array_equal(a["samples_eq"], b["samples_eq"])


def test_samples_eq_reproducible_given_rseed_npool2(monkeypatch):
    a = _run(monkeypatch, 3, npool=2)
    b = _run(monkeypatch, 3, npool=2)
    assert a["logz"] == b["logz"]
    assert np.array_equal(a["samples_eq"], b["samples_eq"])
