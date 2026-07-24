"""Regression coverage for actionable skill-fidelity diagnostics."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from benchflow.agents.install import deploy_skills
from benchflow.agents.registry import AgentConfig


@pytest.mark.asyncio
async def test_skill_catalog_mismatch_reports_expected_source_and_actual(tmp_path):
    """Guards the skill-fidelity diagnostics follow-up to commit e10b5235."""
    env = MagicMock()
    env.upload_dir = AsyncMock()
    env.exec = AsyncMock(
        side_effect=[
            MagicMock(return_code=1, stdout="", stderr="catalog mismatch"),
            MagicMock(
                return_code=0,
                stdout=(
                    "expected_catalog=rails-dev\n"
                    "source_catalog=foreign-global-skill\n"
                    "actual_catalog[/home/agent/.agents/skills]="
                    "foreign-global-skill\n"
                ),
                stderr="",
            ),
        ]
    )
    agent_cfg = AgentConfig(
        name="test-agent",
        install_cmd="true",
        launch_cmd="true",
        skill_paths=["$HOME/.agents/skills"],
    )
    skills_dir = tmp_path / "skills"
    skill = skills_dir / "rails-dev" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text("# Rails development\n")

    with pytest.raises(RuntimeError) as exc_info:
        await deploy_skills(
            env=env,
            task_path=tmp_path,
            skills_dir=skills_dir,
            agent_cfg=agent_cfg,
            sandbox_user="agent",
            agent_cwd="/app",
        )

    message = str(exc_info.value)
    assert "experiment_fidelity/skill_deployment_missing" in message
    assert "expected_catalog=rails-dev" in message
    assert "source_catalog=foreign-global-skill" in message
    assert "actual_catalog[/home/agent/.agents/skills]=foreign-global-skill" in message
    assert env.exec.await_count == 2
    diagnostic_cmd = env.exec.await_args_list[1].args[0]
    assert "find -L /skills" in diagnostic_cmd
    assert "find -L /home/agent/.agents/skills" in diagnostic_cmd
    assert "rm " not in diagnostic_cmd
    assert "ln " not in diagnostic_cmd
