"""Tests for rollout startup uploads."""

from pathlib import Path

import pytest

from benchflow.rollout import _publish_trajectory_for_verifier, _start_env_and_upload


class FakeUploadEnv:
    def __init__(self) -> None:
        self.started = False
        self.exec_calls: list[tuple[str, str | None, int | None]] = []
        self.uploaded_files: list[tuple[Path, str]] = []
        self.uploaded_dirs: list[tuple[Path, str]] = []
        self.uploaded_file_contents: list[tuple[str, str]] = []
        self.downloaded_files: list[tuple[str, str]] = []

    async def start(self, force_build: bool) -> None:
        self.started = force_build is False

    async def exec(
        self, command: str, user: str | None = None, timeout_sec: int | None = None
    ) -> None:
        self.exec_calls.append((command, user, timeout_sec))

    async def upload_file(self, source: Path | str, target: str) -> None:
        source_path = Path(source)
        self.uploaded_files.append((source_path, target))
        self.uploaded_file_contents.append((source_path.read_text(), target))

    async def upload_dir(self, source: Path, target: str) -> None:
        self.uploaded_dirs.append((source, target))

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        self.downloaded_files.append((str(source_path), str(target_path)))


@pytest.mark.asyncio
async def test_start_env_uploads_task_environment_skills(tmp_path: Path) -> None:
    """Guards ENG-88: remote oracle tasks can import bundled task skills."""
    task = tmp_path / "task"
    (task / "environment" / "skills" / "jax-skills").mkdir(parents=True)
    (task / "solution").mkdir(parents=True)
    (task / "instruction.md").write_text("solve\n")
    (task / "solution" / "solve.sh").write_text("echo ok\n")
    env = FakeUploadEnv()
    timing: dict[str, float] = {}

    await _start_env_and_upload(env, task, timing)

    task_skills = task / "environment" / "skills"
    assert env.started is True
    assert (task / "instruction.md", "/instruction.md") in env.uploaded_files
    assert (task_skills, "/app/.agents/skills") in env.uploaded_dirs
    assert (task / "solution", "/solution") in env.uploaded_dirs
    assert "environment_setup" in timing


@pytest.mark.asyncio
async def test_start_env_skips_task_skills_when_gated(tmp_path: Path) -> None:
    """No-skill rollouts must NOT leak environment/skills into /app/skills.

    Skill activation onto the agent's discovery path is handled by
    deploy_skills(); the working-dir copy is gated on upload_task_skills so a
    no-skill run never sees the skill content in /app.
    """
    task = tmp_path / "task"
    (task / "environment" / "skills" / "jax-skills").mkdir(parents=True)
    (task / "instruction.md").write_text("solve\n")
    env = FakeUploadEnv()

    await _start_env_and_upload(env, task, {}, upload_task_skills=False)

    task_skills = task / "environment" / "skills"
    # instruction still uploaded; skills withheld from the working dir.
    assert (task / "instruction.md", "/instruction.md") in env.uploaded_files
    assert (task_skills, "/app/.agents/skills") not in env.uploaded_dirs
    assert not any(target == "/app/.agents/skills" for _, target in env.uploaded_dirs)


@pytest.mark.asyncio
async def test_capture_workspace_tars_excluding_heavy_dirs(tmp_path: Path) -> None:
    """_capture_workspace tars the agent cwd (excluding heavy dirs) and downloads
    the single archive to rollout_dir/artifacts/workspace.tgz before teardown."""
    from benchflow.rollout import Rollout, RolloutConfig
    from benchflow.task.paths import RolloutPaths

    r = Rollout(RolloutConfig(task_path=tmp_path / "task", capture_workspace=True))
    r._env = FakeUploadEnv()
    r._agent_cwd = "/app"
    r._rollout_paths = RolloutPaths(rollout_dir=tmp_path / "rollout")

    await r._capture_workspace()

    cmd, user, _ = r._env.exec_calls[-1]
    assert user == "root"
    assert "tar czf /tmp/bf_workspace.tgz" in cmd
    assert "--exclude=node_modules" in cmd and "--exclude=.git" in cmd
    assert "-C /app ." in cmd
    assert r._env.downloaded_files == [
        ("/tmp/bf_workspace.tgz", str(tmp_path / "rollout" / "artifacts" / "workspace.tgz"))
    ]


@pytest.mark.asyncio
async def test_publish_trajectory_for_verifier_uploads_acp_jsonl() -> None:
    """Guards the skill-eval LLM judge dogfood failure from 2026-05-19."""
    env = FakeUploadEnv()
    trajectory = [{"type": "agent_message", "text": "ok"}]

    await _publish_trajectory_for_verifier(env, trajectory)

    assert ("mkdir -p /logs/agent", "root", 10) in env.exec_calls
    assert env.uploaded_file_contents == [
        (
            '{"type": "agent_message", "text": "ok"}\n',
            "/logs/agent/acp_trajectory.jsonl",
        )
    ]
