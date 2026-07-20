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
    ("model", "key_env", "key", "base_url", "provider", "catalog"),
    [
        (
            "minimax/MiniMax-M3",
            "MINIMAX_API_KEY",
            "mk-test",
            "https://api.minimaxi.com/v1",
            "minimax",
            "minimax-m3.json",
        ),
        (
            "openrouter/minimax/minimax-m3",
            "OPENROUTER_API_KEY",
            "or-test",
            "https://openrouter.ai/api/v1",
            "openrouter",
            "openrouter.json",
        ),
    ],
)
def test_codex_launch_uses_named_responses_provider(
    model,
    key_env,
    key,
    base_url,
    provider,
    catalog,
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
    assert f"-c model_catalog_json=/root/.codex/model-catalogs/{catalog}" in launch


def test_openrouter_qwen_uses_profile_context_and_filtered_loopback_provider():
    """Guards the Qwen3-Coder-Next pioneer commit's Codex launch settings."""
    env = {
        "OPENROUTER_API_KEY": "or-test",
        "BENCHFLOW_PROVIDER_MODEL_CONTEXT_WINDOW": "262144",
        "BENCHFLOW_PROVIDER_REQUEST_FILTER": "omit-reasoning",
    }
    model = "openrouter/qwen/qwen3-coder-next"
    resolve_provider_env(env, model, "codex-acp")

    launch = _apply_provider_agent_launch(
        "codex-acp -c sandbox_mode=workspace-write",
        agent="codex-acp",
        model=model,
        agent_env=env,
        sandbox_user=None,
    )

    assert "-c model_providers.openrouter.base_url=http://127.0.0.1:17891" in launch
    assert "-c model_context_window=262144" in launch
    assert "-c model_catalog_json=/root/.codex/model-catalogs/openrouter.json" in launch
    assert env["BENCHFLOW_PROVIDER_BASE_URL"] == "https://openrouter.ai/api/v1"


def test_openrouter_qwen_maps_profile_context_for_claude_code():
    """Guards the Qwen3-Coder-Next pioneer commit's Claude context settings."""
    env = {
        "OPENROUTER_API_KEY": "or-test",
        "BENCHFLOW_PROVIDER_MODEL_CONTEXT_WINDOW": "262144",
        "BENCHFLOW_PROVIDER_REQUEST_FILTER": "omit-reasoning",
    }
    model = "openrouter/qwen/qwen3-coder-next"

    resolve_provider_env(env, model, "claude-agent-acp")
    launch = _apply_provider_agent_launch(
        "claude-agent-acp",
        agent="claude-agent-acp",
        model=model,
        agent_env=env,
        sandbox_user=None,
    )

    assert launch == "claude-agent-acp"
    assert env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == "262144"
    assert env["ANTHROPIC_BASE_URL"] == "https://openrouter.ai/api"
    assert env["BENCHFLOW_PROVIDER_REQUEST_FILTER"] == "omit-reasoning"


@pytest.mark.parametrize(
    ("model", "agent", "expected_count"),
    [
        ("openrouter/minimax/minimax-m3", "codex-acp", 1),
        ("openrouter/qwen/qwen3-coder-next", "codex-acp", 1),
        ("openrouter/tencent/hy3", "codex-acp", 1),
        ("openrouter/deepseek/deepseek-v4-flash", "codex-acp", 1),
        ("openrouter/deepseek/deepseek-v4-pro", "codex-acp", 1),
        ("openrouter/moonshotai/kimi-k2.6", "codex-acp", 1),
        ("openrouter/z-ai/glm-5.2", "codex-acp", 1),
        ("openrouter/qwen/qwen3.5-397b-a17b", "codex-acp", 1),
        ("openrouter/openai/gpt-oss-120b", "codex-acp", 1),
        ("minimax/MiniMax-M3", "codex-acp", 0),
        ("openrouter/openai/gpt-5", "codex-acp", 0),
        ("openrouter/minimax/MiniMax-M3", "codex-acp", 0),
        ("openrouter/minimax/minimax-m3", "claude-agent-acp", 0),
        ("openrouter/qwen/qwen3-coder-next", "claude-agent-acp", 0),
    ],
)
def test_multi_agent_disable_is_exactly_scoped_to_registered_openrouter_models(
    model,
    agent,
    expected_count,
):
    """Project-supported OpenRouter models run as single Codex agents."""
    launch = _apply_provider_agent_launch(
        agent,
        agent=agent,
        model=model,
        agent_env={"BENCHFLOW_PROVIDER_BASE_URL": "https://example.invalid/v1"},
        sandbox_user=None,
    )

    assert launch.count("-c features.multi_agent=false") == expected_count
    if (
        expected_count
        and agent == "codex-acp"
        and model != "openrouter/minimax/minimax-m3"
    ):
        assert f"-c model={model.removeprefix('openrouter/')}" in launch
        assert "-c sandbox_mode=danger-full-access" in launch


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


@pytest.mark.asyncio
async def test_openrouter_codex_catalog_contains_registered_project_models():
    """The deterministic catalog must cover every project-owned OpenRouter profile."""
    uploaded: dict[str, str] = {}

    class FakeEnv:
        async def exec(self, _cmd, timeout_sec=None):
            return SimpleNamespace(return_code=0, stdout="", stderr="")

        async def upload_file(self, source, destination):
            uploaded[destination] = Path(source).read_text()

    await write_credential_files(
        FakeEnv(),
        "codex-acp",
        {"OPENROUTER_API_KEY": "or-test", "OPENAI_API_KEY": ""},
        AGENTS["codex-acp"],
        "openrouter/qwen/qwen3-coder-next",
        "/root",
    )

    path = "/root/.codex/model-catalogs/openrouter.json"
    catalog = json.loads(uploaded[path])
    qwen = next(
        entry for entry in catalog["models"] if entry["slug"] == "qwen/qwen3-coder-next"
    )
    assert qwen["default_reasoning_level"] == "none"
    assert qwen["supported_reasoning_levels"] == [
        {"effort": "none", "description": "Non-thinking"}
    ]
    assert qwen["supports_reasoning_summaries"] is False

    expected_reasoning = {
        "tencent/hy3": ["none", "low", "high"],
        "deepseek/deepseek-v4-flash": ["high", "xhigh"],
        "deepseek/deepseek-v4-pro": ["high", "xhigh"],
        "z-ai/glm-5.2": ["high", "xhigh"],
        "openai/gpt-oss-120b": ["low", "medium", "high"],
    }
    by_slug = {entry["slug"]: entry for entry in catalog["models"]}
    for slug, efforts in expected_reasoning.items():
        assert [
            item["effort"] for item in by_slug[slug]["supported_reasoning_levels"]
        ] == efforts
        assert by_slug[slug]["supports_reasoning_summaries"] is True

    project_slugs = {
        "qwen/qwen3-coder-next",
        "tencent/hy3",
        "deepseek/deepseek-v4-flash",
        "deepseek/deepseek-v4-pro",
        "moonshotai/kimi-k2.6",
        "z-ai/glm-5.2",
        "qwen/qwen3.5-397b-a17b",
        "openai/gpt-oss-120b",
    }
    for slug in project_slugs:
        instructions = by_slug[slug]["base_instructions"]
        assert "Use only tools declared" in instructions
        assert "exec_command" in instructions
        assert "apply_patch" in instructions
        assert "read_file, list_dir, list_directory, or spawn_agent" in instructions

    for slug in ("moonshotai/kimi-k2.6", "qwen/qwen3.5-397b-a17b"):
        assert [
            item["effort"] for item in by_slug[slug]["supported_reasoning_levels"]
        ] == ["high"]
        assert "provider default" in by_slug[slug]["supported_reasoning_levels"][0][
            "description"
        ]
