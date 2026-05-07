"""Dispatch tests for the per-paper combiner registry."""

import pytest

from dwarfjeans.ingest.combiners import (
    COMBINER_REGISTRY,
    CombinePolicy,
    get_combiner,
)
from dwarfjeans.ingest.combiners import default as default_module

PER_PAPER_BIBCODES = (
    "2017ApJ...838....8L",
    "2018ApJ...857..145L",
    "2022ApJ...939...41C",
    "2023AJ....165...55C",
    "2020ApJ...892..137S",
    "2024ApJ...968...21H",
)


@pytest.mark.parametrize("bibcode", PER_PAPER_BIBCODES)
def test_per_paper_handler_registered(bibcode):
    assert bibcode in COMBINER_REGISTRY
    fn, policy = get_combiner(bibcode)
    # The per-paper handler must NOT be the default module's combine.
    assert fn is not default_module.combine, (
        f"{bibcode!r} dispatched to default; expected the per-paper handler"
    )
    # The handler must live in dwarfjeans.ingest.combiners.<paper>.
    assert fn.__module__.startswith("dwarfjeans.ingest.combiners."), fn.__module__
    assert fn.__module__ != "dwarfjeans.ingest.combiners.default"
    assert isinstance(policy, CombinePolicy)


def test_unknown_bibcode_falls_back_to_default():
    fn, policy = get_combiner("9999XXX...000...00X")
    assert fn is default_module.combine
    assert isinstance(policy, CombinePolicy)
    assert policy.sigma_sys_kms == 0.0
    assert policy.p_threshold == 0.01


def test_registry_covers_all_per_epoch_catalog_bibcodes():
    """The seven per-epoch catalogs in data/star_catalogs cover six
    distinct source paper bibcodes — all six must dispatch to a
    per-paper handler, not the default."""
    expected = set(PER_PAPER_BIBCODES)
    assert set(COMBINER_REGISTRY) >= expected, (
        f"missing per-paper handlers: {expected - set(COMBINER_REGISTRY)}"
    )
