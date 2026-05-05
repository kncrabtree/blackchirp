"""Shared fixtures for the blackchirp test suite.

The tests run against the bundled example data under
``python/example-data/`` (two directories up from this file's
location). Both fixtures are single-segment ``Forever`` acquisitions
with multiple backups, so they exercise the differential-FID API as
well as the schema-loading paths.
"""

from __future__ import annotations

import os

import pytest

from blackchirp import BCExperiment

_EXAMPLE_DATA = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "example-data")
)


def _fixture_path(name: str) -> str:
    return os.path.join(_EXAMPLE_DATA, name)


@pytest.fixture(scope="session")
def mtbe_path() -> str:
    """Filesystem path to the v1-style ``mtbe`` fixture."""
    return _fixture_path("mtbe")


@pytest.fixture(scope="session")
def v2_ftmw_path() -> str:
    """Filesystem path to the v2 FTMW fixture."""
    return _fixture_path("v2-ftmw")


@pytest.fixture
def mtbe_exp(mtbe_path) -> BCExperiment:
    """Freshly loaded ``mtbe`` experiment.

    Re-loaded per test to keep tests independent — some tests mutate
    the FID data in place.
    """
    return BCExperiment(mtbe_path)


@pytest.fixture
def v2_ftmw_exp(v2_ftmw_path) -> BCExperiment:
    """Freshly loaded ``v2-ftmw`` experiment."""
    return BCExperiment(v2_ftmw_path)


@pytest.fixture(params=["mtbe", "v2-ftmw"])
def any_exp(request) -> BCExperiment:
    """Parametrised over both bundled fixtures."""
    return BCExperiment(_fixture_path(request.param))
