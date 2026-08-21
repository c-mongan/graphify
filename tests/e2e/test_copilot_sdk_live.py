"""Opt-in authenticated Copilot SDK smoke test.

Normal CI never runs this test. It uses only a synthetic fixture and the same
Graphify extraction path as a user command.
"""
from __future__ import annotations

import os

import pytest

from graphify.llm import extract_files_direct


pytestmark = pytest.mark.skipif(
    os.environ.get("GRAPHIFY_COPILOT_E2E") != "1",
    reason="set GRAPHIFY_COPILOT_E2E=1 to run the authenticated Copilot test",
)


def test_live_copilot_extracts_synthetic_relationship(tmp_path):
    fixture = tmp_path / "fixture.md"
    fixture.write_text(
        "Component Alpha calls Component Beta through function invoke_beta.\n",
        encoding="utf-8",
    )
    result = extract_files_direct(
        [fixture],
        backend="copilot-sdk",
        model=os.environ.get("GRAPHIFY_COPILOT_E2E_MODEL"),
        root=tmp_path,
    )
    assert result["nodes"]
    assert result["edges"]
    assert any(node.get("source_file") == "fixture.md" for node in result["nodes"])
    assert any(edge.get("source_file") == "fixture.md" for edge in result["edges"])
    if result.get("input_tokens") is not None:
        assert result["input_tokens"] >= 0
    if result.get("output_tokens") is not None:
        assert result["output_tokens"] >= 0
