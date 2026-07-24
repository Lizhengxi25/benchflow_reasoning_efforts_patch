from pathlib import Path
from types import SimpleNamespace

import pytest

from benchflow.rollout import Rollout


class _FakeSandbox:
    def __init__(self, *, tar_return_code: int = 0) -> None:
        self.tar_return_code = tar_return_code
        self.commands: list[str] = []

    async def exec(self, command: str, timeout_sec: int):
        self.commands.append(command)
        if command.startswith("tar "):
            return SimpleNamespace(
                return_code=self.tar_return_code,
                stdout="",
                stderr="tar failed" if self.tar_return_code else "",
            )
        return SimpleNamespace(return_code=0, stdout="", stderr="")

    async def download_file(self, source: str, target: Path) -> None:
        target.write_bytes(b"workspace archive")


@pytest.mark.asyncio
async def test_workspace_debug_capture_downloads_private_archive(tmp_path) -> None:
    """Guards the SkillsBench 1.1 debug-artifact integration."""
    rollout = Rollout.__new__(Rollout)
    rollout._rollout_dir = tmp_path / "rollout"
    rollout._rollout_dir.mkdir()
    rollout._env = _FakeSandbox()
    rollout._agent_cwd = "/workspace"

    await rollout._capture_workspace_artifact()

    archive = rollout._rollout_dir / "artifacts" / "workspace.tgz"
    assert archive.read_bytes() == b"workspace archive"
    assert (archive.stat().st_mode & 0o777) == 0o600
    assert any("-C /workspace ." in command for command in rollout._env.commands)
    assert any("--exclude=node_modules" in command for command in rollout._env.commands)
    assert rollout._env.commands[-1] == "rm -f /tmp/benchflow-workspace.tgz"


@pytest.mark.asyncio
async def test_workspace_debug_capture_persists_nonfatal_error(tmp_path) -> None:
    """Guards capture failure from replacing a valid rollout result."""
    rollout = Rollout.__new__(Rollout)
    rollout._rollout_dir = tmp_path / "rollout"
    rollout._rollout_dir.mkdir()
    rollout._env = _FakeSandbox(tar_return_code=2)
    rollout._agent_cwd = "/workspace"

    await rollout._capture_workspace_artifact()

    artifacts = rollout._rollout_dir / "artifacts"
    assert not (artifacts / "workspace.tgz").exists()
    error = artifacts / "workspace_capture_error.txt"
    assert "workspace tar exited 2" in error.read_text()
    assert (error.stat().st_mode & 0o777) == 0o600
