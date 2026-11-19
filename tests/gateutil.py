"""Shared skip/fail behaviour for the model-validation gate.

Locally, a missing checkpoint or a missing dataset is a skip with an explicit
reason. In CI the workflow sets ``CI_REQUIRE_MODEL_GATE=1``, and the same
condition fails the job. A skip must not be a green model-validation run.
"""

import os

import pytest


def gate_missing(reason: str) -> None:
    if os.environ.get("CI_REQUIRE_MODEL_GATE") == "1":
        pytest.fail(reason)
    pytest.skip(reason)
