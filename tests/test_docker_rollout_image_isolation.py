"""Regression coverage for rollout-scoped Docker build tags."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from benchflow.sandbox._base import ExecResult
from benchflow.sandbox.docker import DockerSandbox
from benchflow.sandbox.process import DockerProcess
from benchflow.task import RolloutPaths, SandboxConfig


def _sandbox(
    tmp_path: Path,
    session_id: str,
    *,
    environment_dir: Path | None = None,
    rollout_dir: Path | None = None,
) -> DockerSandbox:
    environment_dir = environment_dir or tmp_path / session_id / "environment"
    environment_dir.mkdir(parents=True, exist_ok=True)
    (environment_dir / "Dockerfile").write_text("FROM scratch\n")
    rollout_paths = RolloutPaths(rollout_dir) if rollout_dir is not None else None
    if rollout_paths is not None:
        rollout_paths.mkdir()
    return DockerSandbox(
        environment_dir=environment_dir,
        environment_name="shared-task",
        session_id=session_id,
        rollout_paths=rollout_paths,
        task_env_config=SandboxConfig(),
    )


def test_same_session_in_distinct_jobs_gets_distinct_image_tags(tmp_path):
    """Guards the SkillsBench v1.1 collision found after commit e10b5235.

    A collaborator may launch the same rollout name below another jobs root.
    Session-only hashes made those independent sandbox instances share a tag.
    """
    environment_dir = tmp_path / "task" / "environment"
    first = _sandbox(
        tmp_path,
        "same-session",
        environment_dir=environment_dir,
        rollout_dir=tmp_path / "jobs-a" / "run" / "same-session",
    )
    second = _sandbox(
        tmp_path,
        "same-session",
        environment_dir=environment_dir,
        rollout_dir=tmp_path / "jobs-b" / "run" / "same-session",
    )

    assert first._env_vars.main_image_name != second._env_vars.main_image_name


@pytest.mark.asyncio
async def test_same_session_in_distinct_jobs_uses_distinct_compose_projects(tmp_path):
    """Guards the Compose-isolation follow-up to commit e10b5235.

    Image tags alone do not isolate containers and networks. Capture the actual
    host Compose argv, and also verify that the ACP live process reuses the same
    rollout-scoped project selected by DockerSandbox.
    """
    environment_dir = tmp_path / "task" / "environment"
    first = _sandbox(
        tmp_path,
        "same-session",
        environment_dir=environment_dir,
        rollout_dir=tmp_path / "jobs-a" / "run" / "same-session",
    )
    second = _sandbox(
        tmp_path,
        "same-session",
        environment_dir=environment_dir,
        rollout_dir=tmp_path / "jobs-b" / "run" / "same-session",
    )
    third = _sandbox(
        tmp_path,
        "same-session",
        environment_dir=tmp_path / "other-task" / "environment",
        rollout_dir=tmp_path / "jobs-a" / "run" / "same-session",
    )
    commands: list[list[str]] = []

    async def fake_exec(*args, **kwargs):
        del kwargs
        commands.append(list(args))
        process = AsyncMock()
        process.returncode = 0
        process.communicate = AsyncMock(return_value=(b"", b""))
        return process

    with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
        await first._run_docker_compose_command(["ps"])
        await second._run_docker_compose_command(["ps"])
        await third._run_docker_compose_command(["ps"])

    project_names = [
        command[command.index("--project-name") + 1] for command in commands
    ]
    assert project_names == [
        first._compose_project_name,
        second._compose_project_name,
        third._compose_project_name,
    ]
    assert len(set(project_names)) == 3
    assert first._instance_scope_digest in first._env_vars.main_image_name
    assert first._instance_scope_digest in first._compose_project_name
    assert second._instance_scope_digest in second._env_vars.main_image_name
    assert second._instance_scope_digest in second._compose_project_name
    assert third._instance_scope_digest in third._env_vars.main_image_name
    assert third._instance_scope_digest in third._compose_project_name

    first_process = DockerProcess.from_sandbox_env(first)
    second_process = DockerProcess.from_sandbox_env(second)
    third_process = DockerProcess.from_sandbox_env(third)
    assert first_process._compose_cmd()[3] == project_names[0]
    assert second_process._compose_cmd()[3] == project_names[1]
    assert third_process._compose_cmd()[3] == project_names[2]


@pytest.mark.asyncio
async def test_parallel_same_task_rollouts_cannot_overwrite_each_others_image(
    tmp_path,
):
    """Guards the rollout-image isolation follow-up to commit e10b5235.

    The first rollout deliberately pauses after build while the second builds.
    With the old shared ``bf__shared-task`` tag, the first ``compose up`` saw
    the second rollout's image. Rollout-scoped tags keep both references stable.
    """
    first = _sandbox(tmp_path, "shared-task__no-skill")
    second = _sandbox(tmp_path, "shared-task__with-skill")
    assert first._env_vars.main_image_name != second._env_vars.main_image_name

    DockerSandbox._image_build_locks.clear()
    image_store: dict[str, str] = {}
    observed: dict[str, str] = {}
    first_built = asyncio.Event()
    second_built = asyncio.Event()

    async def build(sandbox: DockerSandbox, label: str) -> None:
        image_store[sandbox._env_vars.main_image_name] = label
        (first_built if label == "no-skill" else second_built).set()

    async def first_compose(command, check=True, timeout_sec=None):
        del check, timeout_sec
        if command[0] == "down":
            await second_built.wait()
        return ExecResult(stdout="", stderr="", return_code=0)

    async def second_compose(command, check=True, timeout_sec=None):
        del command, check, timeout_sec
        return ExecResult(stdout="", stderr="", return_code=0)

    async def compose_up(sandbox: DockerSandbox, label: str) -> None:
        observed[label] = image_store[sandbox._env_vars.main_image_name]

    for sandbox, label, compose in (
        (first, "no-skill", first_compose),
        (second, "with-skill", second_compose),
    ):

        async def run_build(s: DockerSandbox = sandbox, value: str = label) -> None:
            await build(s, value)

        async def run_compose_up(
            s: DockerSandbox = sandbox, value: str = label
        ) -> None:
            await compose_up(s, value)

        sandbox._run_pre_compose_hook = AsyncMock()
        sandbox._run_docker_compose_build = AsyncMock(side_effect=run_build)
        sandbox._run_docker_compose_command = AsyncMock(side_effect=compose)
        sandbox._run_docker_compose_up = AsyncMock(side_effect=run_compose_up)
        sandbox.exec = AsyncMock(
            return_value=ExecResult(stdout="", stderr="", return_code=0)
        )

    first_start = asyncio.create_task(first.start(force_build=True))
    await first_built.wait()
    second_start = asyncio.create_task(second.start(force_build=True))
    await asyncio.gather(first_start, second_start)

    assert observed == {"no-skill": "no-skill", "with-skill": "with-skill"}


@pytest.mark.asyncio
async def test_stop_removes_rollout_scoped_build_tag(tmp_path):
    """Guards commit e10b5235's tag follow-up against unbounded accumulation."""
    sandbox = _sandbox(tmp_path, "shared-task__cleanup")
    image_name = sandbox._env_vars.main_image_name
    sandbox._chown_to_host_user = AsyncMock()
    sandbox._run_docker_compose_command = AsyncMock(
        return_value=ExecResult(stdout="", stderr="", return_code=0)
    )
    sandbox._docker_cli = AsyncMock(
        return_value=ExecResult(stdout="", stderr="", return_code=0)
    )

    await sandbox.stop(delete=False)

    sandbox._docker_cli.assert_awaited_once_with(
        ["image", "rm", image_name],
        check=False,
    )


@pytest.mark.asyncio
async def test_stop_never_removes_prebuilt_image(tmp_path):
    """Guards commit e10b5235's follow-up from deleting prebuilt images."""
    sandbox = _sandbox(tmp_path, "shared-task__prebuilt")
    sandbox._use_prebuilt = True
    sandbox._chown_to_host_user = AsyncMock()
    sandbox._run_docker_compose_command = AsyncMock(
        return_value=ExecResult(stdout="", stderr="", return_code=0)
    )
    sandbox._docker_cli = AsyncMock()

    await sandbox.stop(delete=False)

    sandbox._docker_cli.assert_not_awaited()
