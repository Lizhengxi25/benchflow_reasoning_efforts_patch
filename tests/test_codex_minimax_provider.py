"""Codex custom-provider wiring for MiniMax official API and OpenRouter."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchflow.agents.credentials import write_credential_files
from benchflow.agents.env import resolve_provider_env
from benchflow.agents.registry import AGENTS
from benchflow.rollout import _apply_provider_agent_launch


@pytest.mark.parametrize(
    ("model", "key_env", "key", "base_url", "provider"),
    [
        (
            "minimax/MiniMax-M3",
            "MINIMAX_API_KEY",
            "mk-test",
            "https://api.minimaxi.com/v1",
            "minimax",
        ),
        (
            "openrouter/minimax/minimax-m3",
            "OPENROUTER_API_KEY",
            "or-test",
            "https://openrouter.ai/api/v1",
            "openrouter",
        ),
    ],
)
def test_codex_launch_uses_named_responses_provider(
    model,
    key_env,
    key,
    base_url,
    provider,
):
    env = {key_env: key}
    resolve_provider_env(env, model, "codex-acp")

    launch = _apply_provider_agent_launch(
        "codex-acp -c sandbox_mode=workspace-write",
        agent="codex-acp",
        model=model,
        agent_env=env,
        sandbox_user=None,
    )

    assert f"-c model_provider={provider}" in launch
    assert f"-c model_providers.{provider}.base_url={base_url}" in launch
    assert f"-c model_providers.{provider}.env_key={key_env}" in launch
    assert f"-c model_providers.{provider}.wire_api=responses" in launch
    assert "-c model_context_window=1000000" in launch
    assert "-c model_catalog_json=/root/.codex/model-catalogs/minimax-m3.json" in launch


@pytest.mark.parametrize(
    ("model", "agent", "expected_count"),
    [
        ("openrouter/minimax/minimax-m3", "codex-acp", 1),
        ("minimax/MiniMax-M3", "codex-acp", 0),
        ("openrouter/openai/gpt-5", "codex-acp", 0),
        ("openrouter/minimax/MiniMax-M3", "codex-acp", 0),
        ("openrouter/minimax/minimax-m3", "claude-agent-acp", 0),
    ],
)
def test_multi_agent_disable_is_exactly_scoped_to_openrouter_minimax_m3(
    model,
    agent,
    expected_count,
):
    """Guards the M3 namespace workaround added after commit a48bf34."""
    launch = _apply_provider_agent_launch(
        agent,
        agent=agent,
        model=model,
        agent_env={"BENCHFLOW_PROVIDER_BASE_URL": "https://example.invalid/v1"},
        sandbox_user=None,
    )

    assert launch.count("-c features.multi_agent=false") == expected_count


@pytest.mark.asyncio
async def test_minimax_codex_catalog_is_written_to_agent_home():
    uploaded: dict[str, str] = {}

    class FakeEnv:
        async def exec(self, _cmd, timeout_sec=None):
            return SimpleNamespace(return_code=0, stdout="", stderr="")

        async def upload_file(self, source, destination):
            uploaded[destination] = Path(source).read_text()

    await write_credential_files(
        FakeEnv(),
        "codex-acp",
        {"MINIMAX_API_KEY": "mk-test", "OPENAI_API_KEY": ""},
        AGENTS["codex-acp"],
        "minimax/MiniMax-M3",
        "/root",
    )

    path = "/root/.codex/model-catalogs/minimax-m3.json"
    catalog = json.loads(uploaded[path])
    slugs = {entry["slug"] for entry in catalog["models"]}
    assert {"MiniMax-M3", "minimax/minimax-m3"} <= slugs
    direct = next(entry for entry in catalog["models"] if entry["slug"] == "MiniMax-M3")
    assert direct["default_reasoning_level"] == "high"
    assert direct["shell_type"] == "shell_command"
    assert direct["supports_reasoning_summaries"] is True
