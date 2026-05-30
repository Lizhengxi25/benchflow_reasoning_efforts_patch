# Changelog

## [Unreleased]

### Fork: `feat/reasoning-effort` (off upstream 7a479e9, 2026-05-29)

Not yet upstreamed — lives on
[`Lizhengxi25/benchflow_reasoning_efforts_patch`](https://github.com/Lizhengxi25/benchflow_reasoning_efforts_patch)
branch `feat/reasoning-effort`, commit `483d138`.

#### Added

- **`bench run --reasoning-effort`** — surfaces a per-run reasoning effort
  knob for coding agents whose model accepts one. Values
  `minimal|low|medium|high|xhigh` (mirrors the Codex CLI
  `model_reasoning_effort` TOML key) are validated upfront in
  `RolloutConfig.__post_init__`.
- **`AgentConfig.reasoning_effort_flag`** — per-agent launch-arg template
  with `{value}` placeholder; rolls up into the launch command after the
  no-web suffix. `codex-acp` gets `"-c model_reasoning_effort={value}"`;
  every other agent stays empty and raises a clear `ValueError` at
  `Rollout.setup` if `--reasoning-effort` is passed. The bash word-split
  inside the sandbox keeps each `-c key=value` as a separate argv item.
- **`SDK.run(reasoning_effort=...)`** kwarg, plumbed through to the new
  `RolloutConfig.reasoning_effort` field.
- **Codex native-session harvest** (2026-05-30) — `codex-acp`'s ACP stream is
  lossy (drops `read`/`search` tool results and all reasoning), but the
  underlying `codex` core still persists a complete session JSONL to
  `$HOME/.codex/sessions/**/rollout-*.jsonl` inside the sandbox.
  `Rollout._harvest_codex_native_session` copies the newest one out to
  `trajectory/codex_native_session.jsonl` in `cleanup()` (after skill export,
  before `env.stop`), so a downstream judge can read full `function_call` +
  `function_call_output` tool I/O. No-op for non-codex agents; harvest failure
  is logged and never aborts the rollout.

#### Affects

- `src/benchflow/agents/registry.py` — field + `REASONING_EFFORT_VALUES` +
  `normalize_reasoning_effort()` validator.
- `src/benchflow/rollout.py` — `RolloutConfig` field; `_apply_reasoning_effort`
  helper; called at the two `_agent_launch` computation sites (primary
  setup and `connect_as` for additional roles). Also
  `Rollout._harvest_codex_native_session` + its `cleanup()` call.
- `src/benchflow/sdk.py` — kwarg.
- `src/benchflow/cli/main.py` — `bench run` typer Option.

## 0.3.3 — 2026-05-15

### Added

- **Harvey LAB benchmark** — converter, agent shim, and parity validation for 1,251 legal AI tasks (#239).
- **Harvey LAB Claude Sonnet judge** — switched verifier from Gemini to `claude-sonnet-4-6`, matching the original benchmark default (#264).
- **ProgramBench integration** — new benchmark adapter; TB2 removed; `.ref/` migrated to `benchmarks/` (#237).
- **CLI progress output** — `bench eval create` / `bench run` now show progress messages by default (#264).
- **Skill nudge** — optional prompt injection for skill-enhanced agent runs (#207).
- **Self-generated skill mode** for Codex agent (#233).
- **Integration test suite** for ENG-6 + `OPENAI_BASE_URL` inheritance fix (#255).
- **Modal backend support** — Dockerfile compatibility for Modal environments.
- **CITATION.cff** (#246).
- **`AGENTS.md`** — canonical contributor guide; `CLAUDE.md` deprecated (#258).

### Changed

- **Two-field source pattern** for dataset sourcing (#252).
- **Docs overhaul** — synced from www.benchflow.ai; Mintlify config added then orphaned config removed (#259, #257, #226).
- **`uv sync`** for package management (#232).

### Fixed

- Prevent `TypeError` in `metrics.collect_metrics` when reward is `None` (#243).
- Copy eval `requirements.txt` into Docker build context (#245).
- Resolve agent aliases in `bench agent show` and display aliases in `bench agent list` (#251).
- Guard ACP transports against JSON scalar logs (#236).
- Agent timeout reward fallback for Codex (#234).
- Isolate JS agent runtime installs (#231).
- Route Codex ACP through responses API (#224).
- Deploy skills and forward `solution.env` for oracle runs (#223).
- Honor no-internet tasks for agent runs; disable web tools without prompt mutation (#215).
- Propagate `OPENAI_API_KEY` for vllm provider (#3).
- Preserve arrival order of thought/message within flush windows (#214).
- Record user messages and per-turn agent text in ACP trajectory (#745).
- Chown skill-link parent dirs so sandbox user can write into them.
- Dynamic `--rootdir` in `PYTEST_ADDOPTS` based on task workspace.
- Unique env-file path in `DaytonaPtyProcess` to avoid race conditions (#200).

## 0.2.3 — 2026-04-15

### Added

- `benchmarks/tb2_multiturn-claude-haiku45.yaml` — shipped config for the README's TB2 multi-turn Claude result.
- Daytona resource clamping via `BENCHFLOW_DAYTONA_MAX_CPUS` / `MAX_MEMORY_MB`.

### Changed

- Renamed `skillsbench-claude-glm5.yaml` → `skillsbench-claude-glm51.yaml` to match the model ID.
- `codex --login` correction in `docs/getting-started.md`.
- Restricted sdist build to `src/`, `tests/`, and metadata.

### Fixed

- Verifier sandbox hardening follow-ups across several base-image and tooling edge cases.
- Preserve trusted verifier path entries and workspace answer files.
- Redirect oracle output to container log.
- Align YAML path resolution to config file location.

## 0.2.2 — 2026-04-13

### Added

- **Sandbox hardening tiers 1–3** — layered defense (env scrubbing, path lockdown, workspace
  freeze, wider snapshot, oracle privilege drop) blocking F1–F6 red-team findings.
- **`labs/reward-hack-matrix`** — per-trial timeout support and 0.2.2 sweep handoff scripts.

### Fixed

- Multiple sandbox bypass vectors identified in red-team testing.

## 0.2.1 — 2026-04-12

### Added

- **Sandbox hardening on by default** — `sandbox_user` now defaults to `"agent"` (was `None`/root). Blocks conftest-hook and answer-lookup exploit patterns.
- **Path lockdown** — new `sandbox_locked_paths` parameter makes `/solution` and `/tests` read-only before the verifier runs, blocking `.pth`-injection and similar pre-verify tampering.
- **Verifier failure isolation** — agent errors and verifier errors are now stored separately; a crashing verifier no longer masks the agent result.
- **`labs/benchjack-sandbox-hardening`** — cookbook demonstrating three exploit patterns (P1 conftest-hook, P2 answer-lookup, P7 `.pth`-injection) and their defenses.

### Fixed

- **Oracle runs as `sandbox_user`** — oracle agent now respects path lockdown instead of running as root and bypassing it.
- **Multi-endpoint provider routing** — providers with multiple endpoints now route by the agent's native API protocol.
- **Stale API key shadowing subscription auth** — emits a warning when `ANTHROPIC_API_KEY` env var is present alongside `claude login` credentials.
- **pytest `ini`-injection bypass** — closed a verifier hardening edge case.

### Changed

- Version is now single-sourced via `importlib.metadata`; no more duplicate version string in `__init__.py`.
- **User-facing docs** — new `docs/` directory with getting-started guide, CLI reference, architecture overview, task-authoring guide, and labs index. README trimmed; detailed content moved to `docs/`.

## 0.2.0 — 2026-04-09

**First public release.** A near-complete rearchitecture from the 0.1.x era. API surface has changed — assume breaking changes. Future releases will maintain compatibility within the 0.2.x line. 0.1.x users should treat this as a fresh install; see `.dev-docs/sdk-reference.md` for the new SDK.

### Added

- **Multi-agent, multi-provider, multi-auth matrix** — one YAML config, any supported agent × model × provider × auth combination.
- **Subscription auth support** — use `claude login`, `codex --login`, `gemini` OAuth credentials directly. No API keys required for host-based agent workflows.
- **Vertex AI support** — ADC auth for `google-vertex/`, `anthropic-vertex/`, `vertex-zai/` prefixed models.
- **Provider registry** — add a new LLM endpoint via a dict entry in `providers.py`, no code changes.
- **`benchmarks/` directory** with reusable YAML configs and runner scripts for TB2 and SkillsBench.
- **Auto task download** — YAML configs reference datasets as `org/repo/path` (e.g. `harbor-framework/terminal-bench-2`). Repos are cloned on first use and cached under `.cache/datasets/`.
- **`benchflow tasks init`** — scaffold new tasks.
- **`benchflow tasks check`** — validate task structure.
- **`benchflow cleanup`** — delete old sandboxes with `--max-age` filtering (default 24h).
- **Oracle agent support** — run `solution/solve.sh` directly for task validation.
- **Hello-world-task example** for sanity-testing the agent pipeline.
- **Model generation params** via env vars (`BENCHFLOW_TEMPERATURE`, `BENCHFLOW_TOP_P`, `BENCHFLOW_MAX_TOKENS`).
- **OpenClaw ACP shim** with trajectory parsing and skills support.
- **ACP trajectory capture** — full multi-turn agent trajectories via ACP protocol.

### Changed

- **Skill loading** — agent-targeted with proper precedence; auto-distributed from `task.toml` `skills_dir`.
- **`openclaw-gemini` merged** into `openclaw` — provider mode selected at runtime via `BENCHFLOW_PROVIDER_NAME`.

### Fixed

- **API keys leaking in `ps aux`** — env vars now written inside the container instead of passed via Docker exec `-e`.
- **Subscription auth skipped without `-m`** — `benchflow run` without `--model` now checks correctly.
- **ADC credentials break with `sandbox_user`** (#111) — credentials written to sandbox user's home instead of `/root/`.
- **Daytona sandboxes not cleaned up** (#102) — auto-delete after max age.
- **`benchflow cleanup` ignoring `--max-age`** — was deleting everything regardless of age.
- **readline buffer overflow crashes trial** (#98).
- **OpenClaw ACP shim loses tool command text** (#96).
- **OpenClaw ACP shim hardcodes `anthropic/` prefix** (#95) — now routes correctly for Gemini/GLM models.
- **Oracle agent `PermissionError`** writing `agent/oracle.txt` (#91).
- **Oracle path skips `pre_agent_hooks`** (#92) — services now start before oracle runs.
- **Trial data parity with Harbor** (#90) — richer `result.json`, agent logs, per-phase timing.
- **`SDK.run()` `PermissionError`** — `jobs_dir` subdirectories created as root (#88).
- **Partial trajectory lost on timeout** — saved before timeout raises.
- **Redundant `--version` binary check** removed — was wasting 30s per trial.
- **Trajectory fallback** — scrapes agent-native files when ACP `session/update` is empty (#94).
- **`litellm` upgraded to 1.83.0** for CVE-2026-35030; transitive dep security alerts resolved (13 Dependabot alerts closed).

### Deprecated

- `BaseAgent` re-export — planned removal in 0.3.0
- `Trial` re-export — planned removal in 0.3.0
