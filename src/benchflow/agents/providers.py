"""LLM provider registry.

Every provider that benchflow routes models through lives here — both
"custom" providers (like zai/) that need explicit endpoint config, and
"native" providers (like google-vertex/) that agents already support
but we still register so ``is_vertex_model()`` and
``infer_env_key_for_model()`` have a single source of truth.

Adding a new provider is a registry-only change: append one entry to
``PROVIDERS`` below. No new functions, no shim edits, no SDK edits.
``tests/test_registry_invariants.py`` runs contract checks against every
entry — read it for the executable schema.

Required fields
---------------
- ``name``           Must equal the dict key.
- ``base_url``       Primary endpoint URL. May contain ``{placeholder}``
                     tokens that get expanded from env vars via
                     ``url_params``. Empty string is allowed for
                     "user-supplied at runtime" providers (e.g. ``vllm``).
- ``api_protocol``   "anthropic-messages", "openai-completions", or
                     "openai-responses" — the
                     wire protocol the primary ``base_url`` speaks.
- ``auth_type``      "api_key" | "adc" | "aws" | "none".
                     - "api_key": ``auth_env`` **must** be set.
                     - "adc": Application Default Credentials (GCP). The
                       SDK writes the credential file from
                       ``credential_files`` and sets the corresponding env.
                     - "aws": Bedrock API-key auth via
                       ``AWS_BEARER_TOKEN_BEDROCK`` plus region.
                     - "none": no auth.

Common optional fields
----------------------
- ``auth_env``         Env var holding the API key. Must be set iff
                       ``auth_type == "api_key"``.
- ``url_params``       ``{placeholder: ENV_VAR}`` — every placeholder in
                       ``base_url`` (or any ``endpoints`` URL) must have an
                       entry, and every entry must be referenced somewhere.
- ``endpoints``        ``{api_protocol: url}`` for providers that expose
                       multiple protocol surfaces (e.g. zai serves
                       openai-responses, openai-completions, and
                       anthropic-messages). Picked at runtime based on the
                       agent's ``api_protocol``.
- ``models``           Optional list of model metadata dicts (id, name,
                       contextWindow, etc.) consumed by agent shims. ``id``
                       is required and must be unique within the provider.
- ``credential_files`` List of dicts with ``"path"`` and ``"env_source"``
                       (and optional ``"post_env"``) — used by ADC providers
                       to write the credential blob into the container.
- ``agent_auth_env_overrides`` Optional ``{agent_name: ENV_VAR}`` override for
                       the agent-native env var that should receive
                       ``BENCHFLOW_PROVIDER_API_KEY``. Use this when one
                       Anthropic-compatible provider expects ``x-api-key`` and
                       another expects ``Authorization: Bearer`` for the same
                       agent.
- ``agent_env_overrides`` Optional ``{agent_name: {ENV_VAR: value}}`` values
                       that must be forced for a provider-agent pair after
                       generic env mapping.
- ``agent_launch_suffixes`` Optional ``{agent_name: shell_args}`` appended to
                       an agent launch command for provider-native runtime
                       configuration (for example Codex custom providers).
- ``agent_model_launch_suffixes`` Optional
                       ``{agent_name: {full_model_id: shell_args}}`` appended
                       only when both the agent and full, unmodified model ID
                       match exactly.
- ``agent_files``       Optional ``{agent_name: [{path, content}]}`` static
                       config files written into the agent home before launch.

Look at the existing entries below for worked examples:
``zai`` (multi-endpoint, models metadata), ``google-vertex`` (ADC,
credential_files, url_params), ``vllm`` (user-supplied base_url).
"""

from dataclasses import dataclass, field

_MINIMAX_M3_CODEX_CATALOG = """{
  "models": [
    {
      "slug": "MiniMax-M3",
      "display_name": "MiniMax-M3",
      "description": "MiniMax",
      "default_reasoning_level": "high",
      "supported_reasoning_levels": [
        {"effort": "none", "description": "Think-Off"},
        {"effort": "high", "description": "Adaptive Thinking"}
      ],
      "shell_type": "shell_command",
      "visibility": "list",
      "supported_in_api": true,
      "priority": 0,
      "base_instructions": "You are Codex, a coding agent based on MiniMax-M3. You and the user share the same workspace and collaborate to achieve the user's goals.",
      "supports_reasoning_summaries": true,
      "default_reasoning_summary": "none",
      "support_verbosity": false,
      "truncation_policy": {"mode": "bytes", "limit": 10000},
      "supports_parallel_tool_calls": true,
      "experimental_supported_tools": [],
      "input_modalities": ["text", "image"]
    },
    {
      "slug": "minimax/minimax-m3",
      "display_name": "MiniMax-M3 (OpenRouter)",
      "description": "MiniMax via OpenRouter",
      "default_reasoning_level": "high",
      "supported_reasoning_levels": [
        {"effort": "none", "description": "Think-Off"},
        {"effort": "high", "description": "Adaptive Thinking"}
      ],
      "shell_type": "shell_command",
      "visibility": "list",
      "supported_in_api": true,
      "priority": 0,
      "base_instructions": "You are Codex, a coding agent based on MiniMax-M3. You and the user share the same workspace and collaborate to achieve the user's goals.",
      "supports_reasoning_summaries": true,
      "default_reasoning_summary": "none",
      "support_verbosity": false,
      "truncation_policy": {"mode": "bytes", "limit": 10000},
      "supports_parallel_tool_calls": true,
      "experimental_supported_tools": [],
      "input_modalities": ["text", "image"]
    }
  ]
}
"""

_OPENROUTER_CODEX_CATALOG = """{
  "models": [
    {
      "slug": "minimax/minimax-m3",
      "display_name": "MiniMax-M3 (OpenRouter)",
      "description": "MiniMax via OpenRouter",
      "default_reasoning_level": "high",
      "supported_reasoning_levels": [
        {"effort": "none", "description": "Think-Off"},
        {"effort": "high", "description": "Adaptive Thinking"}
      ],
      "shell_type": "shell_command",
      "visibility": "list",
      "supported_in_api": true,
      "priority": 0,
      "base_instructions": "You are Codex, a coding agent based on MiniMax-M3. You and the user share the same workspace and collaborate to achieve the user's goals.",
      "supports_reasoning_summaries": true,
      "default_reasoning_summary": "none",
      "support_verbosity": false,
      "truncation_policy": {"mode": "bytes", "limit": 10000},
      "supports_parallel_tool_calls": true,
      "experimental_supported_tools": [],
      "input_modalities": ["text", "image"]
    },
    {
      "slug": "qwen/qwen3-coder-next",
      "display_name": "Qwen3 Coder Next (OpenRouter)",
      "description": "Qwen3 Coder Next via OpenRouter",
      "default_reasoning_level": "none",
      "supported_reasoning_levels": [
        {"effort": "none", "description": "Non-thinking"}
      ],
      "shell_type": "shell_command",
      "visibility": "list",
      "supported_in_api": true,
      "priority": 0,
      "base_instructions": "You are Codex, a coding agent based on Qwen3-Coder-Next. You and the user share the same workspace and collaborate to achieve the user's goals. Use only tools declared in each API request. Use exec_command for shell commands, file reads, searches, and directory listings, and use apply_patch for edits. Never call read_file, list_dir, list_directory, or spawn_agent. Work in the current agent without subagents.",
      "supports_reasoning_summaries": false,
      "default_reasoning_summary": "none",
      "support_verbosity": false,
      "truncation_policy": {"mode": "bytes", "limit": 10000},
      "supports_parallel_tool_calls": true,
      "experimental_supported_tools": [],
      "input_modalities": ["text"]
    },
    {
      "slug": "tencent/hy3",
      "display_name": "Tencent Hy3 (OpenRouter)",
      "description": "Tencent Hy3 via OpenRouter",
      "default_reasoning_level": "none",
      "supported_reasoning_levels": [
        {"effort": "none", "description": "No thinking"},
        {"effort": "low", "description": "Low reasoning"},
        {"effort": "high", "description": "High reasoning"}
      ],
      "shell_type": "shell_command",
      "visibility": "list",
      "supported_in_api": true,
      "priority": 0,
      "base_instructions": "You are Codex, a coding agent based on Tencent Hy3. You and the user share the same workspace and collaborate to achieve the user's goals. Use only tools declared in each API request. Use exec_command for shell commands, file reads, searches, and directory listings, and use apply_patch for edits. Never call read_file, list_dir, list_directory, or spawn_agent. Work in the current agent without subagents.",
      "supports_reasoning_summaries": true,
      "default_reasoning_summary": "none",
      "support_verbosity": false,
      "truncation_policy": {"mode": "bytes", "limit": 10000},
      "supports_parallel_tool_calls": true,
      "experimental_supported_tools": [],
      "input_modalities": ["text"]
    },
    {
      "slug": "deepseek/deepseek-v4-flash",
      "display_name": "DeepSeek V4 Flash (OpenRouter)",
      "description": "DeepSeek V4 Flash via OpenRouter",
      "default_reasoning_level": "high",
      "supported_reasoning_levels": [
        {"effort": "high", "description": "High reasoning"},
        {"effort": "xhigh", "description": "Maximum reasoning"}
      ],
      "shell_type": "shell_command",
      "visibility": "list",
      "supported_in_api": true,
      "priority": 0,
      "base_instructions": "You are Codex, a coding agent based on DeepSeek V4 Flash. You and the user share the same workspace and collaborate to achieve the user's goals. Use only tools declared in each API request. Use exec_command for shell commands, file reads, searches, and directory listings, and use apply_patch for edits. Never call read_file, list_dir, list_directory, or spawn_agent. Work in the current agent without subagents.",
      "supports_reasoning_summaries": true,
      "default_reasoning_summary": "none",
      "support_verbosity": false,
      "truncation_policy": {"mode": "bytes", "limit": 10000},
      "supports_parallel_tool_calls": true,
      "experimental_supported_tools": [],
      "input_modalities": ["text"]
    },
    {
      "slug": "deepseek/deepseek-v4-pro",
      "display_name": "DeepSeek V4 Pro (OpenRouter)",
      "description": "DeepSeek V4 Pro via OpenRouter",
      "default_reasoning_level": "high",
      "supported_reasoning_levels": [
        {"effort": "high", "description": "High reasoning"},
        {"effort": "xhigh", "description": "Maximum reasoning"}
      ],
      "shell_type": "shell_command",
      "visibility": "list",
      "supported_in_api": true,
      "priority": 0,
      "base_instructions": "You are Codex, a coding agent based on DeepSeek V4 Pro. You and the user share the same workspace and collaborate to achieve the user's goals. Use only tools declared in each API request. Use exec_command for shell commands, file reads, searches, and directory listings, and use apply_patch for edits. Never call read_file, list_dir, list_directory, or spawn_agent. Work in the current agent without subagents.",
      "supports_reasoning_summaries": true,
      "default_reasoning_summary": "none",
      "support_verbosity": false,
      "truncation_policy": {"mode": "bytes", "limit": 10000},
      "supports_parallel_tool_calls": true,
      "experimental_supported_tools": [],
      "input_modalities": ["text"]
    },
    {
      "slug": "moonshotai/kimi-k2.6",
      "display_name": "Kimi K2.6 (OpenRouter)",
      "description": "Kimi K2.6 via OpenRouter",
      "default_reasoning_level": "high",
      "supported_reasoning_levels": [
        {"effort": "high", "description": "Harness placeholder; provider default is preserved"}
      ],
      "shell_type": "shell_command",
      "visibility": "list",
      "supported_in_api": true,
      "priority": 0,
      "base_instructions": "You are Codex, a coding agent based on Kimi K2.6. You and the user share the same workspace and collaborate to achieve the user's goals. Use only tools declared in each API request. Use exec_command for shell commands, file reads, searches, and directory listings, and use apply_patch for edits. Never call read_file, list_dir, list_directory, or spawn_agent. Work in the current agent without subagents.",
      "supports_reasoning_summaries": true,
      "default_reasoning_summary": "none",
      "support_verbosity": false,
      "truncation_policy": {"mode": "bytes", "limit": 10000},
      "supports_parallel_tool_calls": true,
      "experimental_supported_tools": [],
      "input_modalities": ["text", "image"]
    },
    {
      "slug": "z-ai/glm-5.2",
      "display_name": "GLM 5.2 (OpenRouter)",
      "description": "GLM 5.2 via OpenRouter",
      "default_reasoning_level": "high",
      "supported_reasoning_levels": [
        {"effort": "high", "description": "High reasoning"},
        {"effort": "xhigh", "description": "Maximum reasoning"}
      ],
      "shell_type": "shell_command",
      "visibility": "list",
      "supported_in_api": true,
      "priority": 0,
      "base_instructions": "You are Codex, a coding agent based on GLM 5.2. You and the user share the same workspace and collaborate to achieve the user's goals. Use only tools declared in each API request. Use exec_command for shell commands, file reads, searches, and directory listings, and use apply_patch for edits. Never call read_file, list_dir, list_directory, or spawn_agent. Work in the current agent without subagents.",
      "supports_reasoning_summaries": true,
      "default_reasoning_summary": "none",
      "support_verbosity": false,
      "truncation_policy": {"mode": "bytes", "limit": 10000},
      "supports_parallel_tool_calls": true,
      "experimental_supported_tools": [],
      "input_modalities": ["text"]
    },
    {
      "slug": "qwen/qwen3.5-397b-a17b",
      "display_name": "Qwen3.5 397B A17B (OpenRouter)",
      "description": "Qwen3.5 397B A17B via OpenRouter",
      "default_reasoning_level": "high",
      "supported_reasoning_levels": [
        {"effort": "high", "description": "Harness placeholder; provider default is preserved"}
      ],
      "shell_type": "shell_command",
      "visibility": "list",
      "supported_in_api": true,
      "priority": 0,
      "base_instructions": "You are Codex, a coding agent based on Qwen3.5 397B A17B. You and the user share the same workspace and collaborate to achieve the user's goals. Use only tools declared in each API request. Use exec_command for shell commands, file reads, searches, and directory listings, and use apply_patch for edits. Never call read_file, list_dir, list_directory, or spawn_agent. Work in the current agent without subagents.",
      "supports_reasoning_summaries": true,
      "default_reasoning_summary": "none",
      "support_verbosity": false,
      "truncation_policy": {"mode": "bytes", "limit": 10000},
      "supports_parallel_tool_calls": true,
      "experimental_supported_tools": [],
      "input_modalities": ["text", "image"]
    },
    {
      "slug": "openai/gpt-oss-120b",
      "display_name": "GPT-OSS-120B (OpenRouter)",
      "description": "GPT-OSS-120B via OpenRouter",
      "default_reasoning_level": "medium",
      "supported_reasoning_levels": [
        {"effort": "low", "description": "Low reasoning"},
        {"effort": "medium", "description": "Medium reasoning"},
        {"effort": "high", "description": "High reasoning"}
      ],
      "shell_type": "shell_command",
      "visibility": "list",
      "supported_in_api": true,
      "priority": 0,
      "base_instructions": "You are Codex, a coding agent based on GPT-OSS-120B. You and the user share the same workspace and collaborate to achieve the user's goals. Use only tools declared in each API request. Use exec_command for shell commands, file reads, searches, and directory listings, and use apply_patch for edits. Never call read_file, list_dir, list_directory, or spawn_agent. Work in the current agent without subagents.",
      "supports_reasoning_summaries": true,
      "default_reasoning_summary": "none",
      "support_verbosity": false,
      "truncation_policy": {"mode": "bytes", "limit": 10000},
      "supports_parallel_tool_calls": true,
      "experimental_supported_tools": [],
      "input_modalities": ["text"]
    }
  ]
}
"""

_OPENROUTER_CODEX_SINGLE_AGENT_SUFFIXES = {
    # ACP creates the Codex session before set_model, so pin the catalog slug
    # here. The outer no-network task container is the sandbox boundary; its
    # ARM image has no bwrap for Codex's nested workspace sandbox.
    # MiniMax keeps its pre-profile context override for direct BenchFlow use.
    "openrouter/minimax/minimax-m3": (
        "-c model_context_window=1000000 -c features.multi_agent=false"
    ),
    **{
        model: (
            "-c model={model} -c sandbox_mode=danger-full-access "
            "-c features.multi_agent=false"
        )
        for model in (
            "openrouter/qwen/qwen3-coder-next",
            "openrouter/tencent/hy3",
            "openrouter/deepseek/deepseek-v4-flash",
            "openrouter/deepseek/deepseek-v4-pro",
            "openrouter/moonshotai/kimi-k2.6",
            "openrouter/z-ai/glm-5.2",
            "openrouter/qwen/qwen3.5-397b-a17b",
            "openrouter/openai/gpt-oss-120b",
        )
    },
}


@dataclass
class ProviderConfig:
    """Configuration for a custom LLM provider."""

    name: str
    base_url: (
        str  # primary endpoint; may contain {placeholders} expanded via url_params
    )
    api_protocol: str  # protocol for base_url: "openai-responses" | "openai-completions" | "anthropic-messages"
    auth_type: str  # "api_key" | "adc" | "aws" | "none"
    auth_env: str | None = None  # env var holding the API key (None for ADC)
    url_params: dict[str, str] = field(default_factory=dict)  # {placeholder: ENV_VAR}
    models: list[dict] = field(default_factory=list)  # model metadata for agents
    # Multi-protocol support: {protocol: base_url} for providers with multiple APIs.
    # base_url + api_protocol is the primary; endpoints adds alternatives.
    endpoints: dict[str, str] = field(default_factory=dict)
    credential_files: list[dict] = field(default_factory=list)
    # Files to write into container (e.g. GCP ADC).
    # Each dict: {"path": str, "env_source": str, "post_env": {k: v} (optional)}
    agent_auth_env_overrides: dict[str, str] = field(default_factory=dict)
    # Provider-specific destination for BENCHFLOW_PROVIDER_API_KEY per agent.
    agent_env_overrides: dict[str, dict[str, str]] = field(default_factory=dict)
    # Provider-specific fixed env values per agent.
    agent_launch_suffixes: dict[str, str] = field(default_factory=dict)
    # Provider-specific launch arguments. Supports {base_url}, {home}, and {model}.
    agent_model_launch_suffixes: dict[str, dict[str, str]] = field(
        default_factory=dict
    )
    # Exact-model launch arguments keyed by agent, then full unstripped model ID.
    # Templates support the same placeholders as agent_launch_suffixes.
    agent_files: dict[str, list[dict[str, str]]] = field(default_factory=dict)
    # Static files written before launch. Paths support the {home} placeholder.

    @property
    def all_endpoints(self) -> dict[str, str]:
        """Merged view: endpoints dict with base_url/api_protocol as fallback."""
        merged = {self.api_protocol: self.base_url}
        merged.update(self.endpoints)
        return merged


# ── Provider registry ──

PROVIDERS: dict[str, ProviderConfig] = {
    # ── Native Vertex AI providers (agents support these natively) ──
    "google-vertex": ProviderConfig(
        name="google-vertex",
        base_url="https://aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}",
        api_protocol="openai-completions",
        auth_type="adc",
        url_params={
            "project_id": "GOOGLE_CLOUD_PROJECT",
            "location": "GOOGLE_CLOUD_LOCATION",
        },
        credential_files=[
            {
                "path": "{home}/.config/gcloud/application_default_credentials.json",
                "env_source": "GOOGLE_APPLICATION_CREDENTIALS_JSON",
                "post_env": {
                    "GOOGLE_APPLICATION_CREDENTIALS": "{home}/.config/gcloud/application_default_credentials.json",
                },
            }
        ],
    ),
    "anthropic-vertex": ProviderConfig(
        name="anthropic-vertex",
        base_url="https://aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}",
        api_protocol="anthropic-messages",
        auth_type="adc",
        url_params={
            "project_id": "GOOGLE_CLOUD_PROJECT",
            "location": "GOOGLE_CLOUD_LOCATION",
        },
        credential_files=[
            {
                "path": "{home}/.config/gcloud/application_default_credentials.json",
                "env_source": "GOOGLE_APPLICATION_CREDENTIALS_JSON",
                "post_env": {
                    "GOOGLE_APPLICATION_CREDENTIALS": "{home}/.config/gcloud/application_default_credentials.json",
                },
            }
        ],
    ),
    # ── OpenAI-compatible inference servers (user-supplied base_url) ──
    "vllm": ProviderConfig(
        name="vllm",
        base_url="",  # user-supplied via --agent-env BENCHFLOW_PROVIDER_BASE_URL=...
        api_protocol="openai-completions",
        auth_type="api_key",
        auth_env="OPENAI_API_KEY",  # vLLM uses OpenAI-compatible auth
    ),
    "aws-bedrock": ProviderConfig(
        name="aws-bedrock",
        base_url="",  # local Bedrock proxy supplies the runtime URL later
        api_protocol="openai-responses",
        auth_type="aws",
        endpoints={
            "anthropic-messages": "",
        },
    ),
    # ── Custom providers (need explicit endpoint config in agent shims) ──
    "zai": ProviderConfig(
        name="zai",
        base_url="https://api.z.ai/api/paas/v4",
        api_protocol="openai-completions",
        auth_type="api_key",
        auth_env="ZAI_API_KEY",
        endpoints={
            "openai-completions": "https://api.z.ai/api/paas/v4",
            "openai-responses": "https://api.z.ai/api/paas/v4",
            "anthropic-messages": "https://api.z.ai/api/anthropic",
        },
        models=[
            {
                "id": "glm-5",
                "name": "GLM-5",
                "reasoning": True,
                "input": ["text"],
                "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                "contextWindow": 200000,
                "maxTokens": 131072,
            },
            {
                "id": "glm-5.1",
                "name": "GLM-5.1",
                "reasoning": True,
                "input": ["text"],
                "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                "contextWindow": 200000,
                "maxTokens": 131072,
            },
        ],
    ),
    "minimax": ProviderConfig(
        name="minimax",
        base_url="https://api.minimaxi.com/anthropic",
        api_protocol="anthropic-messages",
        auth_type="api_key",
        auth_env="MINIMAX_API_KEY",
        endpoints={
            "anthropic-messages": "https://api.minimaxi.com/anthropic",
            "openai-completions": "https://api.minimaxi.com/v1",
            "openai-responses": "https://api.minimaxi.com/v1",
        },
        agent_auth_env_overrides={
            "claude-agent-acp": "ANTHROPIC_API_KEY",
            "codex-acp": "MINIMAX_API_KEY",
        },
        agent_env_overrides={
            # Codex uses the custom provider below. Empty OpenAI vars prevent
            # inherited OpenAI credentials/base URLs from selecting the native
            # provider or producing a conflicting ~/.codex/auth.json.
            "codex-acp": {"OPENAI_API_KEY": "", "OPENAI_BASE_URL": ""},
        },
        agent_launch_suffixes={
            "codex-acp": (
                "-c model_provider=minimax "
                "-c model_providers.minimax.name=MiniMax "
                "-c model_providers.minimax.base_url={base_url} "
                "-c model_providers.minimax.env_key=MINIMAX_API_KEY "
                "-c model_providers.minimax.wire_api=responses "
                "-c model_context_window=1000000 "
                "-c model_catalog_json={home}/.codex/model-catalogs/minimax-m3.json"
            ),
        },
        agent_files={
            "codex-acp": [
                {
                    "path": "{home}/.codex/model-catalogs/minimax-m3.json",
                    "content": _MINIMAX_M3_CODEX_CATALOG,
                }
            ],
        },
        models=[
            {
                "id": "MiniMax-M3",
                "name": "MiniMax-M3",
                "reasoning": True,
                "input": ["text", "image", "video"],
                "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                "contextWindow": 1000000,
                "maxTokens": 1000000,
            },
            {
                "id": "MiniMax-M2.7",
                "name": "MiniMax-M2.7",
                "reasoning": True,
                "input": ["text"],
                "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                "contextWindow": 204800,
                "maxTokens": 204800,
            },
            {
                "id": "MiniMax-M2.7-highspeed",
                "name": "MiniMax-M2.7 Highspeed",
                "reasoning": True,
                "input": ["text"],
                "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                "contextWindow": 204800,
                "maxTokens": 204800,
            },
            {
                "id": "MiniMax-M2.5",
                "name": "MiniMax-M2.5",
                "reasoning": True,
                "input": ["text"],
                "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                "contextWindow": 204800,
                "maxTokens": 204800,
            },
            {
                "id": "MiniMax-M2.5-highspeed",
                "name": "MiniMax-M2.5 Highspeed",
                "reasoning": True,
                "input": ["text"],
                "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                "contextWindow": 204800,
                "maxTokens": 204800,
            },
            {
                "id": "MiniMax-M2.1",
                "name": "MiniMax-M2.1",
                "reasoning": True,
                "input": ["text"],
                "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                "contextWindow": 204800,
                "maxTokens": 204800,
            },
            {
                "id": "MiniMax-M2.1-highspeed",
                "name": "MiniMax-M2.1 Highspeed",
                "reasoning": True,
                "input": ["text"],
                "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                "contextWindow": 204800,
                "maxTokens": 204800,
            },
            {
                "id": "MiniMax-M2",
                "name": "MiniMax-M2",
                "reasoning": True,
                "input": ["text"],
                "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                "contextWindow": 204800,
                "maxTokens": 204800,
            },
        ],
    ),
    "openrouter": ProviderConfig(
        name="openrouter",
        base_url="https://openrouter.ai/api",
        api_protocol="anthropic-messages",
        auth_type="api_key",
        auth_env="OPENROUTER_API_KEY",
        endpoints={
            "anthropic-messages": "https://openrouter.ai/api",
            "openai-completions": "https://openrouter.ai/api/v1",
            "openai-responses": "https://openrouter.ai/api/v1",
        },
        agent_env_overrides={
            # OpenRouter's Claude Code integration requires this to be present
            # and empty, otherwise Claude Code may prefer Anthropic auth.
            "claude-agent-acp": {"ANTHROPIC_API_KEY": ""},
            # Codex is configured as a named custom provider below. Do not let
            # inherited OpenAI settings override that route.
            "codex-acp": {"OPENAI_API_KEY": "", "OPENAI_BASE_URL": ""},
        },
        agent_auth_env_overrides={
            "codex-acp": "OPENROUTER_API_KEY",
        },
        agent_launch_suffixes={
            "codex-acp": (
                "-c model_provider=openrouter "
                "-c model_providers.openrouter.name=OpenRouter "
                "-c model_providers.openrouter.base_url={base_url} "
                "-c model_providers.openrouter.env_key=OPENROUTER_API_KEY "
                "-c model_providers.openrouter.wire_api=responses "
                "-c model_catalog_json={home}/.codex/model-catalogs/openrouter.json"
            ),
        },
        agent_model_launch_suffixes={
            "codex-acp": _OPENROUTER_CODEX_SINGLE_AGENT_SUFFIXES,
        },
        agent_files={
            "codex-acp": [
                {
                    "path": "{home}/.codex/model-catalogs/openrouter.json",
                    "content": _OPENROUTER_CODEX_CATALOG,
                }
            ],
        },
    ),
}


def find_provider(model: str) -> tuple[str, ProviderConfig] | None:
    """Find the custom provider for a model ID based on its prefix.

    Returns (provider_name, config) or None if no custom provider matches.
    Matches longest prefix first to handle nested prefixes (e.g. google-vertex/ vs google/).
    """
    m = model.lower()
    # Sort by prefix length descending so longer prefixes match first
    candidates = []
    for name, cfg in PROVIDERS.items():
        prefix = f"{name}/"
        if m.startswith(prefix):
            candidates.append((len(prefix), name, cfg))
    if not candidates:
        return None
    candidates.sort(reverse=True, key=lambda x: x[0])
    _, name, cfg = candidates[0]
    return name, cfg


def resolve_base_url(
    provider: ProviderConfig,
    env: dict[str, str],
    protocol: str | None = None,
) -> str:
    """Expand {placeholders} in a provider's base_url using env vars.

    If *protocol* is given and the provider has an ``endpoints`` entry for it,
    that URL is used instead of the primary ``base_url``.

    Raises KeyError if a required env var is missing.
    """
    url = provider.base_url
    if protocol and provider.endpoints.get(protocol):
        url = provider.endpoints[protocol]
    if not provider.url_params:
        return url
    replacements = {}
    for placeholder, env_var in provider.url_params.items():
        value = env.get(env_var)
        if not value:
            raise KeyError(
                f"Provider {provider.name!r} requires {env_var} for "
                f"{{{placeholder}}} in base_url, but it is not set."
            )
        replacements[placeholder] = value
    return url.format_map(replacements)


def strip_provider_prefix(model: str) -> str:
    """Strip a *registered* provider prefix. Unregistered inputs pass through.

    "anthropic-vertex/claude-sonnet-4-6" → "claude-sonnet-4-6"
    "zai/glm-5" → "glm-5"
    "vllm/Qwen/Qwen3-Coder" → "Qwen/Qwen3-Coder"  (HF org/model kept intact)
    "Qwen/Qwen3-Coder" → "Qwen/Qwen3-Coder"       (no registered prefix → unchanged)
    "claude-sonnet-4-6" → "claude-sonnet-4-6"

    Single normalization point for downstream callers (ACP set_model,
    BENCHFLOW_PROVIDER_MODEL env var, Harbor YAML parse). If a model ID
    reaches an agent launcher still prefixed, fix the routing into this
    function — do NOT strip again at the call site. See PRs #154 and #155
    for the symptomatic-patch anti-pattern that caused the original bug.
    """
    result = find_provider(model)
    if result:
        return model[len(result[0]) + 1 :]
    return model


def resolve_auth_env(model: str) -> str | None:
    """Return the env var name needed for a model's provider, or None.

    Returns None for ADC-based, AWS-auth providers, and unknown models.
    """
    result = find_provider(model)
    if result is None:
        return None
    _, cfg = result
    if cfg.auth_type in {"adc", "aws", "none"}:
        return None
    return cfg.auth_env
