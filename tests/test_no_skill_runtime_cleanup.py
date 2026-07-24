"""Regression coverage for base-image skill cleanup in no-skill rollouts."""

import os
import pwd
import shlex
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import benchflow.rollout as rollout_module
from benchflow.agents.registry import AgentConfig
from benchflow.rollout import Role, Rollout, RolloutConfig
from benchflow.rollout._no_skill_cleanup import build_no_skill_cleanup_cmd
from benchflow.rollout._setup import (
    _clear_no_skill_agent_skill_paths,
    _clear_no_skill_skill_roots,
    _no_skill_agent_discovery_targets,
)
from benchflow.skill_policy import SKILL_MODE_NO_SKILL, SKILL_MODE_WITH_SKILL


def _rollout_for_start(mode: str, tmp_path: Path) -> Rollout:
    rollout = Rollout.__new__(Rollout)
    rollout._env = SimpleNamespace(sandbox_id=None)
    rollout._env_externally_owned = False
    rollout._config = SimpleNamespace(
        task_path=tmp_path / "task",
        pre_agent_hooks=[],
        environment_manifest=None,
    )
    rollout._timing = {}
    rollout._task = SimpleNamespace(
        config=SimpleNamespace(
            environment=SimpleNamespace(
                skills_dir="/opt/benchflow/task-skills",
            )
        )
    )
    rollout._environment = None
    rollout._task_skill_policy = SimpleNamespace(mode=mode)
    rollout._agent_cwd = "/app"
    rollout._phase = "setup"
    return rollout


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "expected_calls"),
    [
        (SKILL_MODE_NO_SKILL, 1),
        (SKILL_MODE_WITH_SKILL, 0),
    ],
)
async def test_start_clears_base_image_skills_only_for_no_skill(
    monkeypatch,
    tmp_path,
    mode,
    expected_calls,
):
    """Guards the cleanup follow-up to commit e10b5235 after the 20260724 run."""
    rollout = _rollout_for_start(mode, tmp_path)
    start_env = AsyncMock()
    healthcheck = AsyncMock()
    setup_commands = AsyncMock()
    resolve_cwd = AsyncMock(return_value="/app")
    clear_skills = AsyncMock()
    monkeypatch.setattr(rollout_module, "_start_env_and_upload", start_env)
    monkeypatch.setattr(rollout_module, "_run_environment_healthcheck", healthcheck)
    monkeypatch.setattr(
        rollout_module, "_run_environment_setup_commands", setup_commands
    )
    monkeypatch.setattr(rollout_module, "_resolve_agent_cwd", resolve_cwd)
    monkeypatch.setattr(rollout_module, "_clear_no_skill_skill_roots", clear_skills)

    await rollout.start()

    assert clear_skills.await_count == expected_calls
    if mode == SKILL_MODE_NO_SKILL:
        clear_skills.assert_awaited_once_with(
            rollout._env,
            "/opt/benchflow/task-skills",
            workspace="/app",
        )


@pytest.mark.asyncio
async def test_no_skill_cleanup_resets_default_and_safe_declared_roots():
    """Guards the cleanup follow-up to commit e10b5235 inside the sandbox."""
    env = MagicMock()
    env.exec = AsyncMock(return_value=MagicMock(return_code=0, stdout="", stderr=""))

    await _clear_no_skill_skill_roots(
        env,
        "/opt/benchflow/task-skills",
        workspace="/app",
    )

    command = env.exec.await_args.args[0]
    script = shlex.split(command)[-1]
    assert (
        "validate_target /skills '' '' reset /proc/self/mountinfo "
        "/app /workspace /home /root /output /outputs /oracle /solution "
        "/verifier /tests /testbed_verify /app"
    ) in script
    assert (
        "validate_target /opt/benchflow/task-skills '' '' reset /proc/self/mountinfo"
    ) in script
    assert "reset_target /skills '' '' reset" in script
    assert ("reset_target /opt/benchflow/task-skills '' '' reset") in script
    assert "/proc/self/mountinfo" in command
    assert env.exec.await_args.kwargs["user"] == "root"


@pytest.mark.asyncio
async def test_no_skill_cleanup_refuses_workspace_skills_directory():
    """Guards commit e10b5235's follow-up against deleting ``/app/skills``."""
    env = MagicMock()
    env.exec = AsyncMock()

    with pytest.raises(
        ValueError,
        match="experiment_fidelity/unsafe_skill_cleanup_root",
    ):
        await _clear_no_skill_skill_roots(
            env,
            "/app/skills",
            workspace="/app",
        )

    env.exec.assert_not_awaited()


def _mountinfo_with_only_root(tmp_path: Path) -> Path:
    mountinfo = tmp_path / "mountinfo"
    mountinfo.write_text("1 0 0:1 / / rw,relatime - overlay overlay rw\n")
    return mountinfo


def _run_cleanup(
    targets: list[tuple[str, str | None, str | None, str]],
    *,
    protected: list[str] | None = None,
    mountinfo: Path,
) -> subprocess.CompletedProcess[str]:
    command = build_no_skill_cleanup_cmd(
        targets,
        protected_roots=protected or (),
        mountinfo_path=str(mountinfo),
    )
    return subprocess.run(
        ["/bin/sh", "-c", command],
        capture_output=True,
        text=True,
        check=False,
    )


def test_no_skill_discovery_cleanup_removes_catalog_recreated_by_home_copy(tmp_path):
    """Guards the post-install cleanup follow-up to commit e10b5235."""
    root = tmp_path.resolve()
    home = root / "home" / "agent"
    discovery = home / ".claude" / "skills"
    skill = discovery / "inherited" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text("# inherited\n")

    result = _run_cleanup(
        [(str(discovery), str(home), None, "reset")],
        mountinfo=_mountinfo_with_only_root(root),
    )

    assert result.returncode == 0, result.stderr
    assert discovery.is_dir()
    assert list(discovery.iterdir()) == []
    assert f"cleared_no_skill_root={discovery}" in result.stdout


def test_no_skill_cleanup_refuses_intermediate_symlink_into_workspace(tmp_path):
    """Guards the symlink hardening follow-up to commit e10b5235."""
    root = tmp_path.resolve()
    workspace = root / "app"
    protected_skill = workspace / "task-skills" / "kept" / "SKILL.md"
    protected_skill.parent.mkdir(parents=True)
    protected_skill.write_text("# keep\n")
    opt = root / "opt"
    opt.mkdir()
    (opt / "benchflow").symlink_to(workspace, target_is_directory=True)
    candidate = opt / "benchflow" / "task-skills"

    result = _run_cleanup(
        [(str(candidate), None, None, "reset")],
        protected=[str(workspace)],
        mountinfo=_mountinfo_with_only_root(root),
    )

    assert result.returncode == 86
    assert "intermediate path component is a symlink" in result.stderr
    assert protected_skill.read_text() == "# keep\n"


def test_no_skill_cleanup_refuses_mounted_root_without_deleting_it(tmp_path):
    """Guards the bind-mount hardening follow-up to commit e10b5235."""
    root = tmp_path.resolve()
    candidate = root / "opt" / "task-skills"
    skill = candidate / "mounted" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text("# mounted\n")
    mountinfo = root / "mountinfo-mounted"
    mountinfo.write_text(
        "1 0 0:1 / / rw,relatime - overlay overlay rw\n"
        f"2 1 0:2 / {candidate} rw,relatime - ext4 /dev/mock rw\n"
    )

    result = _run_cleanup(
        [(str(candidate), None, None, "reset")],
        mountinfo=mountinfo,
    )

    assert result.returncode == 86
    assert "cleanup overlaps mounted path" in result.stderr
    assert skill.read_text() == "# mounted\n"


def test_no_skill_cleanup_retains_existing_workspace_discovery_directory(tmp_path):
    """Guards workspace data integrity in the follow-up to commit e10b5235."""
    root = tmp_path.resolve()
    workspace = root / "app"
    discovery = workspace / ".agents" / "skills"
    skill = discovery / "task-fixture" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text("# task fixture\n")

    result = _run_cleanup(
        [(str(discovery), str(workspace), None, "unlink-only")],
        mountinfo=_mountinfo_with_only_root(root),
    )

    assert result.returncode == 86
    assert "existing workspace discovery path is task-owned data" in result.stderr
    assert skill.read_text() == "# task fixture\n"


def test_no_skill_cleanup_validates_all_owners_before_any_deletion(tmp_path):
    """Guards all-or-nothing preflight in the follow-up to commit e10b5235."""
    root = tmp_path.resolve()
    first = root / "opt" / "first-skills"
    sentinel = first / "kept" / "SKILL.md"
    sentinel.parent.mkdir(parents=True)
    sentinel.write_text("# keep\n")
    second = root / "opt" / "second-skills"

    result = _run_cleanup(
        [
            (str(first), None, None, "reset"),
            (
                str(second),
                str(root / "opt"),
                "_benchflow_missing_cleanup_owner_",
                "reset",
            ),
        ],
        mountinfo=_mountinfo_with_only_root(root),
    )

    assert result.returncode == 86
    assert "cleanup owner does not exist" in result.stderr
    assert sentinel.read_text() == "# keep\n"


def test_no_skill_cleanup_owns_new_home_discovery_parents(tmp_path):
    """Guards parent ownership in the follow-up to commit e10b5235."""
    root = tmp_path.resolve()
    home = root / "home"
    home.mkdir()
    discovery = home / ".pi" / "agent" / "skills"
    current = pwd.getpwuid(os.getuid())

    result = _run_cleanup(
        [(str(discovery), str(home), current.pw_name, "reset")],
        mountinfo=_mountinfo_with_only_root(root),
    )

    assert result.returncode == 0, result.stderr
    for created in (home / ".pi", home / ".pi" / "agent", discovery):
        stat = created.stat()
        assert stat.st_uid == current.pw_uid
        assert stat.st_gid == current.pw_gid


def test_no_skill_cleanup_owns_a_new_home_anchor(tmp_path):
    """Guards the SkillsBench v1.1 home-anchor bug found after commit e10b5235."""
    root = tmp_path.resolve()
    home = root / "nonstandard-home" / "agent"
    discovery = home / ".claude" / "skills"
    current = pwd.getpwuid(os.getuid())
    chown_log = root / "chown.log"
    command = build_no_skill_cleanup_cmd(
        [(str(discovery), str(home), current.pw_name, "reset")],
        mountinfo_path=str(_mountinfo_with_only_root(root)),
    )

    command_bin = root / "bin-with-recording-chown"
    command_bin.mkdir()
    for executable in ("readlink", "rm", "mkdir", "find", "id"):
        resolved = shutil.which(executable)
        assert resolved is not None
        (command_bin / executable).symlink_to(resolved)
    chown = command_bin / "chown"
    chown.write_text('#!/bin/sh\nprintf \'%s\\n\' "$2" >> "$CHOWN_LOG"\n')
    chown.chmod(0o755)

    result = subprocess.run(
        ["/bin/sh", "-c", command],
        env={"PATH": str(command_bin), "CHOWN_LOG": str(chown_log)},
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    owned_paths = chown_log.read_text().splitlines()
    assert str(home) in owned_paths
    assert str(discovery) in owned_paths
    assert discovery.is_dir()


def test_no_skill_cleanup_refuses_overlapping_targets(tmp_path):
    """Guards target preflight in the follow-up to commit e10b5235."""
    root = tmp_path.resolve()
    parent = root / "home" / ".agents"
    child = parent / "skills"

    result = _run_cleanup(
        [
            (str(parent), str(root / "home"), None, "reset"),
            (str(child), str(root / "home"), None, "unlink-only"),
        ],
        mountinfo=_mountinfo_with_only_root(root),
    )

    assert result.returncode == 86
    assert "cleanup target overlaps another target" in result.stderr
    assert not parent.exists()


def test_no_skill_cleanup_runs_without_python_on_path(tmp_path):
    """Guards dependency-light cleanup in the follow-up to commit e10b5235."""
    root = tmp_path.resolve()
    candidate = root / "opt" / "task-skills"
    skill = candidate / "kept" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text("# keep\n")
    command = build_no_skill_cleanup_cmd(
        [(str(candidate), None, None, "reset")],
        mountinfo_path=str(_mountinfo_with_only_root(root)),
    )

    command_bin = root / "bin"
    command_bin.mkdir()
    for executable in ("readlink", "rm", "mkdir", "find", "id", "chown"):
        resolved = shutil.which(executable)
        assert resolved is not None
        (command_bin / executable).symlink_to(resolved)
    result = subprocess.run(
        ["/bin/sh", "-c", command],
        env={"PATH": str(command_bin)},
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert candidate.is_dir()
    assert list(candidate.iterdir()) == []


@pytest.mark.asyncio
async def test_no_skill_agent_cleanup_expands_selected_discovery_paths():
    """Guards the selected-agent cleanup follow-up to commit e10b5235."""
    env = MagicMock()
    env.exec = AsyncMock(return_value=MagicMock(return_code=0, stdout="", stderr=""))
    agent_cfg = AgentConfig(
        name="claude-agent-acp",
        install_cmd="true",
        launch_cmd="true",
        skill_paths=["$HOME/.claude/skills", "$WORKSPACE/.agents/skills"],
    )

    await _clear_no_skill_agent_skill_paths(
        env,
        agent_cfg,
        sandbox_user="agent",
        workspace="/app",
    )

    script = shlex.split(env.exec.await_args.args[0])[-1]
    assert (
        "validate_target /home/agent/.claude/skills /home/agent "
        "agent reset /proc/self/mountinfo"
    ) in script
    assert (
        "validate_target /app/.agents/skills /app agent unlink-only "
        "/proc/self/mountinfo"
    ) in script


def test_no_skill_discovery_duplicate_uses_workspace_safe_policy():
    """Guards strict target merging in the follow-up to commit e10b5235."""
    agent_cfg = AgentConfig(
        name="overlapping-agent",
        install_cmd="true",
        launch_cmd="true",
        skill_paths=["$HOME/.agents/skills", "$WORKSPACE/.agents/skills"],
    )

    targets = _no_skill_agent_discovery_targets(
        agent_cfg,
        sandbox_user=None,
        workspace="/root",
    )

    assert targets == (("/root/.agents/skills", "/root", None, "unlink-only"),)


def _connect_ready_rollout(
    tmp_path: Path,
    *,
    primary_agent: str,
    primary_cfg: AgentConfig | None,
) -> Rollout:
    rollout = Rollout.__new__(Rollout)
    rollout._config = RolloutConfig(
        task_path=tmp_path / "task",
        agent=primary_agent,
        model="test-model",
        environment="docker",
        sandbox_user="agent",
    )
    rollout._rollout_dir = tmp_path
    rollout._rollout_name = "no-skill-regression"
    rollout._agent_env = {}
    rollout._agent_launch = primary_agent
    rollout._agent_cwd = "/app"
    rollout._env = SimpleNamespace()
    rollout._usage_runtime = None
    rollout._required_skill_names = ()
    rollout._timing = {}
    rollout._task_skill_policy = SimpleNamespace(mode=SKILL_MODE_NO_SKILL)
    rollout._agent_cfg = primary_cfg
    rollout._reapply_ask_user_handler = lambda: None
    rollout._attach_trajectory_writer = lambda _rollout_dir: None
    rollout._task = None
    return rollout


@pytest.mark.asyncio
async def test_primary_connect_cleans_registry_paths_when_install_was_skipped(
    monkeypatch,
    tmp_path,
):
    """Guards the skip-install cleanup follow-up to commit e10b5235."""
    cfg = AgentConfig(
        name="claude-agent-acp",
        install_cmd="true",
        launch_cmd="true",
        skill_paths=["$HOME/.claude/skills"],
    )
    rollout = _connect_ready_rollout(
        tmp_path,
        primary_agent="claude-agent-acp",
        primary_cfg=None,
    )
    calls: list[str] = []

    async def clear_discovery(*args, **kwargs):
        calls.append("clear")
        assert args[1] is cfg

    async def connect_acp(**kwargs):
        calls.append("connect")
        return (AsyncMock(), AsyncMock(), AsyncMock(), "claude-agent-acp")

    monkeypatch.setattr(
        rollout_module,
        "_clear_no_skill_agent_skill_paths",
        clear_discovery,
    )
    rollout._planes = SimpleNamespace(
        agent_config=MagicMock(return_value=cfg),
        ensure_litellm_runtime=AsyncMock(
            side_effect=lambda **kwargs: (kwargs["agent_env"], None)
        ),
        connect_acp=connect_acp,
    )

    await rollout.connect()

    assert calls == ["clear", "connect"]
    rollout._planes.agent_config.assert_called_once_with("claude-agent-acp")


@pytest.mark.asyncio
async def test_connect_as_cleans_after_deferred_role_install_before_connect(
    monkeypatch,
    tmp_path,
):
    """Guards the scene-role cleanup follow-up to commit e10b5235."""
    primary_cfg = AgentConfig(
        name="codex-acp",
        install_cmd="true",
        launch_cmd="true",
        skill_paths=["$HOME/.agents/skills"],
    )
    role_cfg = AgentConfig(
        name="claude-agent-acp",
        install_cmd="true",
        launch_cmd="true",
        skill_paths=["$HOME/.claude/skills"],
    )
    rollout = _connect_ready_rollout(
        tmp_path,
        primary_agent="codex-acp",
        primary_cfg=primary_cfg,
    )
    rollout._disallow_web_tools = False
    calls: list[str] = []

    async def install_agent(*args, **kwargs):
        calls.append("install")
        return role_cfg

    async def clear_discovery(*args, **kwargs):
        calls.append("clear")
        assert args[1] is role_cfg

    async def connect_acp(**kwargs):
        calls.append("connect")
        return (AsyncMock(), AsyncMock(), AsyncMock(), "claude-agent-acp")

    monkeypatch.setattr(
        rollout_module,
        "_clear_no_skill_agent_skill_paths",
        clear_discovery,
    )
    rollout._planes = SimpleNamespace(
        agent_launch=lambda agent, *, disallow_web_tools: agent,
        agent_config=MagicMock(return_value=role_cfg),
        resolve_agent_env=lambda agent, model, env: dict(env or {}),
        ensure_litellm_runtime=AsyncMock(
            side_effect=lambda **kwargs: (kwargs["agent_env"], None)
        ),
        install_agent=install_agent,
        write_credential_files=AsyncMock(),
        upload_subscription_auth=AsyncMock(),
        apply_web_tool_policy=AsyncMock(),
        connect_acp=connect_acp,
    )
    role = Role(
        name="solver",
        agent="claude-agent-acp",
        model="test-model",
    )

    await rollout.connect_as(role)

    assert calls == ["install", "clear", "connect"]
