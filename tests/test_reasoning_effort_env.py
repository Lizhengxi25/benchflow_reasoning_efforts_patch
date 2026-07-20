"""Env-driven reasoning effort (Claude Code MAX_THINKING_TOKENS path)."""

import pytest

from benchflow.agents.env import resolve_agent_env
from benchflow.agents.registry import CLAUDE_CODE_THINKING_BUDGETS
from benchflow.rollout import _apply_reasoning_effort, _apply_reasoning_effort_env


def test_claude_agent_acp_launch_untouched_by_reasoning_effort():
    launch = _apply_reasoning_effort("agent acp", "claude-agent-acp", "high")
    assert launch == "agent acp"


@pytest.mark.parametrize("effort,budget", sorted(CLAUDE_CODE_THINKING_BUDGETS.items()))
def test_claude_agent_acp_injects_thinking_budget(effort, budget):
    env = _apply_reasoning_effort_env({}, "claude-agent-acp", effort)
    assert env == {"MAX_THINKING_TOKENS": budget}


def test_no_effort_leaves_env_unchanged():
    env = {"EXISTING": "1"}
    assert _apply_reasoning_effort_env(env, "claude-agent-acp", None) is env
    assert env == {"EXISTING": "1"}


def test_flag_based_agent_skips_env_injection():
    assert _apply_reasoning_effort_env({}, "codex-acp", "high") == {}


def test_unsupported_agent_still_raises_and_names_env_agents():
    with pytest.raises(ValueError) as exc:
        _apply_reasoning_effort("x", "pi-acp", "high")
    assert "claude-agent-acp" in str(exc.value)
    assert "codex-acp" in str(exc.value)


def test_haiku_model_caps_claude_code_output_tokens():
    env = resolve_agent_env(
        "claude-agent-acp", "claude-haiku-4-5", {"ANTHROPIC_API_KEY": "k"}
    )
    assert env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == "64000"
    assert env["ANTHROPIC_MODEL"] == "claude-haiku-4-5"
    assert "ANTHROPIC_BASE_URL" not in env


def test_non_haiku_model_keeps_128k_output_tokens():
    env = resolve_agent_env(
        "claude-agent-acp", "claude-sonnet-4-6", {"ANTHROPIC_API_KEY": "k"}
    )
    assert env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == "128000"


def test_model_aliases_pinned_to_rollout_model():
    env = resolve_agent_env(
        "claude-agent-acp", "claude-haiku-4-5", {"ANTHROPIC_API_KEY": "k"}
    )
    for alias in ("SONNET", "OPUS", "HAIKU"):
        assert env[f"ANTHROPIC_DEFAULT_{alias}_MODEL"] == "claude-haiku-4-5"


def test_explicit_alias_override_wins():
    env = resolve_agent_env(
        "claude-agent-acp",
        "claude-haiku-4-5",
        {"ANTHROPIC_API_KEY": "k", "ANTHROPIC_DEFAULT_SONNET_MODEL": "claude-sonnet-4-6"},
    )
    assert env["ANTHROPIC_DEFAULT_SONNET_MODEL"] == "claude-sonnet-4-6"
