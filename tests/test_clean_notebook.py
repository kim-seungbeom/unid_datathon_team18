"""Use constructed JSON only; never load executed user notebooks."""

import json

import pytest

from scripts.check_staged import CLEAN_NOTEBOOK, blocked_path, clean_notebook


def notebook_bytes(**changes):
    cell = {
        "cell_type": "code",
        "source": "print('synthetic')",
        "metadata": {},
        "outputs": [],
        "execution_count": None,
        **changes,
    }
    return json.dumps({"nbformat": 4, "metadata": {}, "cells": [cell]}).encode()


def test_clean_notebook_requires_explicit_exception():
    assert blocked_path(CLEAN_NOTEBOOK)
    assert blocked_path("other.ipynb")
    assert blocked_path("outputs/features.npy")
    assert clean_notebook(notebook_bytes())


@pytest.mark.parametrize(
    "changes",
    [
        {"outputs": [{"output_type": "stream", "text": "synthetic output"}]},
        {"execution_count": 0},
        {"attachments": {"synthetic.png": {"image/png": "synthetic"}}},
        {"metadata": {"execution": {"synthetic": True}}},
        {"source": "data:image/png;base64,synthetic"},
    ],
)
def test_reject_executed_notebook_or_attachments(changes):
    assert not clean_notebook(notebook_bytes(**changes))


@pytest.mark.parametrize("content", [b"not json", b"null", b"{}", b'{"cells": [null]}'])
def test_reject_invalid_notebook(content):
    assert not clean_notebook(content)
