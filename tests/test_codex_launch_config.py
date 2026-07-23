import json

import pytest

from benchflow.acp.runtime import _apply_launch_owned_agent_config
from benchflow.agents.registry import AGENTS


def _profile() -> dict:
    return {
        "display_name": "GLM 5.2 (OpenRouter)",
        "description": "GLM 5.2 via OpenRouter",
        "default_reasoning_level": "high",
        "supported_reasoning_levels": [
            {"effort": "high", "description": "High reasoning"},
            {"effort": "xhigh", "description": "Maximum reasoning"},
        ],
        "base_instructions": "Complete the assigned coding task.",
        "supports_reasoning_summaries": True,
        "input_modalities": ["text"],
    }


def test_codex_native_model_and_reasoning_are_launch_overrides():
    launch = _apply_launch_owned_agent_config(
        agent="codex-acp",
        agent_launch="codex-acp",
        agent_env={},
        model="gpt-5.5",
        reasoning_effort="xhigh",
    )

    assert launch == "codex-acp -c model=gpt-5.5 -c model_reasoning_effort=xhigh"


def test_gpt55_launch_preserves_workspace_write_and_adds_model_and_reasoning():
    launch = _apply_launch_owned_agent_config(
        agent="codex-acp",
        agent_launch=AGENTS["codex-acp"].launch_cmd,
        agent_env={},
        model="gpt-5.5",
        reasoning_effort="xhigh",
    )

    assert "-c sandbox_mode=workspace-write" in launch
    assert launch.endswith("-c model=gpt-5.5 -c model_reasoning_effort=xhigh")
    assert "tools.web_search=false" not in launch


def test_codex_openrouter_profile_materializes_alias_catalog_and_context():
    env = {
        "CODEX_CONFIG": json.dumps({"model": "benchflow-openrouter-z-ai-glm-5.2"}),
        "BENCHFLOW_CODEX_MODEL_PROFILE_JSON": json.dumps(_profile()),
        "BENCHFLOW_PROVIDER_MODEL_CONTEXT_WINDOW": "1048576",
        "BENCHFLOW_CODEX_SANDBOX_MODE": "danger-full-access",
    }

    launch = _apply_launch_owned_agent_config(
        agent="codex-acp",
        agent_launch="codex-acp",
        agent_env=env,
        model="openrouter/z-ai/glm-5.2",
        reasoning_effort="high",
    )

    assert "-c model=benchflow-openrouter-z-ai-glm-5.2" in launch
    assert "-c model_reasoning_effort=high" in launch
    assert "-c model_context_window=1048576" in launch
    assert "-c sandbox_mode=danger-full-access" in launch
    assert launch.rfind("sandbox_mode=danger-full-access") > launch.rfind(
        "sandbox_mode=workspace-write"
    )
    assert 'model_catalog_json="$h/.codex/benchflow-model-catalog.json"' in launch

    catalog = json.loads(env["BENCHFLOW_CODEX_MODEL_CATALOG_JSON"])
    entry = catalog["models"][0]
    assert entry["slug"] == "benchflow-openrouter-z-ai-glm-5.2"
    assert entry["default_reasoning_level"] == "high"
    assert [item["effort"] for item in entry["supported_reasoning_levels"]] == [
        "high",
        "xhigh",
    ]
    assert "multi_agent" not in launch


def test_codex_profile_rejects_invalid_context_before_process_start():
    env = {
        "CODEX_CONFIG": '{"model":"benchflow-openrouter-z-ai-glm-5.2"}',
        "BENCHFLOW_CODEX_MODEL_PROFILE_JSON": json.dumps(_profile()),
        "BENCHFLOW_PROVIDER_MODEL_CONTEXT_WINDOW": "not-an-int",
    }

    with pytest.raises(ValueError, match="positive integer"):
        _apply_launch_owned_agent_config(
            agent="codex-acp",
            agent_launch="codex-acp",
            agent_env=env,
            model="openrouter/z-ai/glm-5.2",
            reasoning_effort=None,
        )
