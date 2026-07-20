import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchflow.acp.session import ACPSession
from benchflow.acp.types import StopReason
from benchflow.rollout import (
    _count_codex_native_task_completions,
    _prefer_acp_with_codex_native_fallback,
)


@pytest.mark.asyncio
async def test_native_completion_count_targets_session_and_sums_matches() -> None:
    env = SimpleNamespace(
        exec=AsyncMock(return_value=SimpleNamespace(stdout="1\n2\nnot-a-count\n"))
    )

    count = await _count_codex_native_task_completions(
        env, "019f-test-session", None
    )

    assert count == 3
    command = env.exec.await_args.args[0]
    assert "/root/.codex/sessions" in command
    assert "*019f-test-session.jsonl" in command
    assert '"type":"task_complete"' in command
    assert env.exec.await_args.kwargs == {"user": "root", "timeout_sec": 10}


@pytest.mark.asyncio
async def test_acp_result_wins_and_cancels_native_watcher() -> None:
    watcher_cancelled = asyncio.Event()

    async def execute():
        return [{"type": "agent_message", "text": "done"}], 0

    async def watch_native():
        try:
            await asyncio.Future()
        finally:
            watcher_cancelled.set()

    client = SimpleNamespace(cancel=AsyncMock())
    session = ACPSession("session-id")

    result = await _prefer_acp_with_codex_native_fallback(
        execute(), watch_native(), client, session
    )

    assert result == ([{"type": "agent_message", "text": "done"}], 0)
    assert watcher_cancelled.is_set()
    client.cancel.assert_not_awaited()


@pytest.mark.asyncio
async def test_native_completion_recovers_stuck_single_turn_acp() -> None:
    execute_cancelled = asyncio.Event()
    session = ACPSession("session-id")

    async def execute():
        session.record_user_prompt("solve it")
        session.handle_update(
            {
                "sessionUpdate": "agent_message_chunk",
                "content": {"type": "text", "text": "finished"},
            }
        )
        try:
            await asyncio.Future()
        finally:
            execute_cancelled.set()

    async def watch_native():
        return None

    client = SimpleNamespace(cancel=AsyncMock())

    trajectory, tool_count = await _prefer_acp_with_codex_native_fallback(
        execute(),
        watch_native(),
        client,
        session,
        completion_grace_sec=0.01,
        cancel_grace_sec=0.01,
    )

    assert execute_cancelled.is_set()
    client.cancel.assert_awaited_once()
    assert session.stop_reason == StopReason.END_TURN
    assert tool_count == 0
    assert trajectory == [
        {"type": "user_message", "text": "solve it"},
        {"type": "agent_message", "text": "finished"},
    ]


@pytest.mark.asyncio
async def test_external_cancellation_drains_both_fallback_tasks() -> None:
    execute_started = asyncio.Event()
    execute_cancelled = asyncio.Event()
    watcher_started = asyncio.Event()
    watcher_cancelled = asyncio.Event()

    async def execute():
        execute_started.set()
        try:
            await asyncio.Future()
        finally:
            execute_cancelled.set()

    async def watch_native():
        watcher_started.set()
        try:
            await asyncio.Future()
        finally:
            watcher_cancelled.set()

    task = asyncio.create_task(
        _prefer_acp_with_codex_native_fallback(
            execute(),
            watch_native(),
            SimpleNamespace(cancel=AsyncMock()),
            ACPSession("session-id"),
        )
    )
    await asyncio.gather(execute_started.wait(), watcher_started.wait())
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert execute_cancelled.is_set()
    assert watcher_cancelled.is_set()
