"""Gated live guard for the Claude ACP version used by the v1.1 leaderboard.

Skipped by default. Run with ``RUN_ACP_DEP_GUARD=1`` (needs ``git``, ``npm``,
``node``, and network):

    RUN_ACP_DEP_GUARD=1 uv run --extra dev python -m pytest \
        tests/test_acp_pinned_protocol_guard.py -q

It builds the exact Zed source revision whose SDK bundles Claude Code 2.1.19,
then verifies the legacy ``session/set_model`` path used for provider aliases.
No credentials are needed.
"""

import asyncio
import contextlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_ACP_DEP_GUARD") != "1",
    reason=(
        "gated live ACP guard; set RUN_ACP_DEP_GUARD=1 "
        "(needs git + npm + node + network)"
    ),
)

PINNED_SOURCE_REV = "670fb18728514c367cf600925c475bb2bd123914"
EXPECTED_ADAPTER_VERSION = "0.13.1"
EXPECTED_SDK_VERSION = "0.2.19"
EXPECTED_CLAUDE_CODE_VERSION = "2.1.19"


def _tool_or_skip(name: str) -> str:
    path = shutil.which(name)
    if not path:
        pytest.skip(f"{name} not available")
    return path


async def _probe_legacy_agent(entry: Path) -> tuple[set[str], dict]:
    from benchflow.acp.client import ACPClient
    from benchflow.acp.transport import StdioTransport

    client = ACPClient(StdioTransport("node", [str(entry)], env={}, cwd="/tmp"))
    try:
        await client.connect()
        await asyncio.wait_for(client.initialize(), timeout=60)
        await asyncio.wait_for(client.session_new(cwd="/tmp"), timeout=90)
        opts = client.session.config_options or []
        option_ids = {
            o["id"]
            for o in opts
            if isinstance(o, dict) and isinstance(o.get("id"), str)
        }
        result = await asyncio.wait_for(
            client.set_model("provider-model-probe"), timeout=60
        )
        return option_ids, result
    finally:
        with contextlib.suppress(Exception):
            await client.close()


def test_pinned_claude_acp_matches_v11_leaderboard_runtime(tmp_path):
    """Guards SkillsBench 7dcfb802 and Zed 670fb187's ACP runtime contract."""
    git = _tool_or_skip("git")
    npm = _tool_or_skip("npm")
    _tool_or_skip("node")
    source = tmp_path / "claude-code-acp"
    subprocess.run(
        [
            git,
            "clone",
            "--filter=blob:none",
            "--no-checkout",
            "https://github.com/zed-industries/claude-agent-acp.git",
            str(source),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
    )
    subprocess.run(
        [git, "-C", str(source), "checkout", "--detach", PINNED_SOURCE_REV],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    subprocess.run(
        [npm, "ci", "--no-audit", "--no-fund"],
        cwd=source,
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
    )
    subprocess.run(
        [npm, "run", "build"],
        cwd=source,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    entry = source / "dist" / "index.js"
    assert entry.is_file(), f"pinned agent entry not found: {entry}"

    ids, set_model_result = asyncio.run(_probe_legacy_agent(entry))
    assert ids == set()
    assert set_model_result == {}

    adapter_metadata = json.loads((source / "package.json").read_text())
    assert adapter_metadata["name"] == "@zed-industries/claude-code-acp"
    assert adapter_metadata["version"] == EXPECTED_ADAPTER_VERSION
    sdk_package = (
        source / "node_modules" / "@anthropic-ai" / "claude-agent-sdk" / "package.json"
    )
    sdk_metadata = json.loads(sdk_package.read_text())
    assert sdk_metadata["version"] == EXPECTED_SDK_VERSION
    assert sdk_metadata["claudeCodeVersion"] == EXPECTED_CLAUDE_CODE_VERSION
    cli_source = (sdk_package.parent / "cli.js").read_text()
    assert cli_source.count('if(lK("tengu_bash_haiku_prefetch",!0)){') == 1
    assert 'if(!1&&lK("tengu_bash_haiku_prefetch",!0)){' not in cli_source
