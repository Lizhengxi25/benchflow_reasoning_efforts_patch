"""Tests for ACP client ↔ mock agent — Step 10."""

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from benchflow.acp.client import ACPClient, ACPError
from benchflow.acp.container_transport import ContainerTransport
from benchflow.acp.session import ACPSession
from benchflow.acp.transport import StdioTransport
from benchflow.acp.types import StopReason, ToolCallStatus

MOCK_AGENT = str(Path(__file__).parent / "fixtures" / "mock_acp_agent.py")
MOCK_AGENT_INTERLEAVED = str(
    Path(__file__).parent / "fixtures" / "mock_acp_agent_interleaved.py"
)


class TestACPClient:
    @pytest.mark.asyncio
    async def test_initialize(self) -> None:
        client = ACPClient(StdioTransport(sys.executable, [MOCK_AGENT]))
        try:
            await client.connect()
            result = await client.initialize()
            assert result.protocol_version == 0
            assert result.agent_info is not None
            assert result.agent_info.name == "mock-agent"
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_session_new(self) -> None:
        client = ACPClient(StdioTransport(sys.executable, [MOCK_AGENT]))
        try:
            await client.connect()
            await client.initialize()
            session = await client.session_new()
            assert session.session_id == "mock-session-1"
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_prompt_and_response(self) -> None:
        client = ACPClient(StdioTransport(sys.executable, [MOCK_AGENT]))
        try:
            await client.connect()
            await client.initialize()
            await client.session_new()

            result = await client.prompt("Hello, agent!")
            assert result.stop_reason == StopReason.END_TURN

            session = client.session
            assert session is not None
            assert "I received: Hello, agent!" in session.full_message
            assert len(session.tool_calls) == 1
            assert session.tool_calls[0].kind == "bash"
            assert session.tool_calls[0].title == "echo hello"
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_from_config_stdio(self) -> None:
        client = ACPClient.from_config(
            command=f"{sys.executable} {MOCK_AGENT}",
            transport_type="stdio",
        )
        try:
            await client.connect()
            result = await client.initialize()
            assert result.agent_info.name == "mock-agent"
        finally:
            await client.close()

    def test_from_config_missing_command(self) -> None:
        with pytest.raises(ValueError, match="command required"):
            ACPClient.from_config(transport_type="stdio")

    def test_from_config_unknown_transport(self) -> None:
        with pytest.raises(ValueError, match="Unknown transport"):
            ACPClient.from_config(transport_type="sse")

    @pytest.mark.asyncio
    async def test_set_model(self) -> None:
        client = ACPClient(StdioTransport(sys.executable, [MOCK_AGENT]))
        try:
            await client.connect()
            await client.initialize()
            await client.session_new()
            # Mock agent returns error for unknown methods, but set_model
            # should raise ACPError for the unknown method response
            with pytest.raises(ACPError):
                await client.set_model("some-model")
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_error_response_preserves_structured_data(self) -> None:
        """Guards the diagnostic-loss regression from commit dea87821."""
        transport = AsyncMock()
        transport.receive.return_value = {
            "jsonrpc": "2.0",
            "id": 100001,
            "error": {
                "code": -32603,
                "message": "Internal error",
                "data": {
                    "message": "Provider overloaded",
                    "codex_error_info": "other",
                },
            },
        }
        client = ACPClient(transport)

        with pytest.raises(ACPError) as exc_info:
            await client._read_until_response(100001)

        error = exc_info.value
        assert error.code == -32603
        assert error.message == "Internal error"
        assert error.data == {
            "message": "Provider overloaded",
            "codex_error_info": "other",
        }
        assert error.detail_message == "Provider overloaded"
        # Existing callers and retry classifiers keep the same string contract.
        assert str(error) == "ACP error -32603: Internal error"

    def test_acp_error_constructor_remains_backwards_compatible(self) -> None:
        """Guards existing two-argument callers after commit dea87821."""
        error = ACPError(400, "Bad request")

        assert error.data is None
        assert error.detail_message == "Bad request"
        assert str(error) == "ACP error 400: Bad request"

    @pytest.mark.asyncio
    async def test_prompt_without_session_raises(self) -> None:
        client = ACPClient(StdioTransport(sys.executable, [MOCK_AGENT]))
        try:
            await client.connect()
            await client.initialize()
            with pytest.raises(RuntimeError, match="No active session"):
                await client.prompt("hello")
        finally:
            await client.close()


class TestACPSession:
    def test_handle_tool_call(self):
        session = ACPSession("test-session")
        session.handle_update(
            {
                "sessionUpdate": "tool_call",
                "toolCallId": "tc_1",
                "title": "echo hello",
                "kind": "bash",
            }
        )
        assert len(session.tool_calls) == 1
        assert session.tool_calls[0].tool_call_id == "tc_1"
        assert session.tool_calls[0].kind == "bash"

    def test_handle_tool_call_update(self):
        session = ACPSession("test-session")
        session.handle_update(
            {
                "sessionUpdate": "tool_call",
                "toolCallId": "tc_1",
                "title": "echo",
                "kind": "bash",
            }
        )
        session.handle_update(
            {
                "sessionUpdate": "tool_call_update",
                "toolCallId": "tc_1",
                "status": "completed",
            }
        )
        assert session.tool_calls[0].status == ToolCallStatus.COMPLETED

    def test_handle_invalid_tool_call_status(self):
        """Invalid status should fall back to IN_PROGRESS, not crash."""
        session = ACPSession("test-session")
        session.handle_update(
            {
                "sessionUpdate": "tool_call",
                "toolCallId": "tc_1",
                "title": "echo",
                "kind": "bash",
            }
        )
        # Should not raise — invalid status handled gracefully
        session.handle_update(
            {
                "sessionUpdate": "tool_call_update",
                "toolCallId": "tc_1",
                "status": "totally_invalid_status",
            }
        )
        assert session.tool_calls[0].status == ToolCallStatus.IN_PROGRESS

    def test_handle_message_chunks(self):
        session = ACPSession("test-session")
        session.handle_update(
            {
                "sessionUpdate": "agent_message_chunk",
                "content": {"type": "text", "text": "Hello "},
            }
        )
        session.handle_update(
            {
                "sessionUpdate": "agent_message_chunk",
                "content": {"type": "text", "text": "world"},
            }
        )
        assert session.full_message == "Hello world"

    def test_handle_thought_chunks(self):
        session = ACPSession("test-session")
        session.handle_update(
            {
                "sessionUpdate": "agent_thought_chunk",
                "content": {"type": "text", "text": "thinking..."},
            }
        )
        assert session.full_thought == "thinking..."

    def test_handle_unknown_update_type(self):
        """Unknown update types should be silently ignored."""
        session = ACPSession("test-session")
        # Should not raise
        session.handle_update({"sessionUpdate": "unknown_type"})
        assert len(session.tool_calls) == 0

    def test_tool_call_update_for_unknown_id(self):
        """Update for non-existent tool call auto-creates a record.

        Agents like Gemini CLI skip the initial tool_call notification and
        send only tool_call_update, so the session synthesizes a record.
        """
        session = ACPSession("test-session")
        session.handle_update(
            {
                "sessionUpdate": "tool_call_update",
                "toolCallId": "nonexistent",
                "status": "completed",
            }
        )
        assert len(session.tool_calls) == 1
        assert session.tool_calls[0].tool_call_id == "nonexistent"
        assert session.tool_calls[0].kind == "tool"


class TestStdioTransportOversizedLine:
    """StdioTransport must recover from oversized lines (LimitOverrunError)."""

    @pytest.mark.asyncio
    async def test_stdio_transport_drain_oversized_line(self) -> None:
        """Oversized line on stdout is skipped; following JSON-RPC returns normally.

        Feeds the oversized chunk first, lets receive() hit LimitOverrunError and
        call drain_oversized_line (which clears the buffer), then feeds the next
        valid JSON-RPC line so readline() can find it. This matches the real
        ordering (stdin -> drain -> next line) that a live process would produce.
        """
        limit = 64
        reader = asyncio.StreamReader(limit=limit)
        # Oversized chunk WITH a newline: triggers "Separator is found, but
        # chunk is longer than limit" on readline().
        reader.feed_data(b"x" * (limit * 3) + b"\n")

        transport = StdioTransport(sys.executable, [])
        fake_process = MagicMock()
        fake_process.stdout = reader
        fake_process.stdin = MagicMock()
        transport._process = fake_process

        async def feed_valid_later() -> None:
            # Give receive() time to drain, then supply the next line.
            # drain_oversized_line clears the buffer and consumes up to the
            # next \n (the oversized newline was flushed with the clear), so
            # we feed a dummy newline to satisfy drain's readuntil, then the
            # real JSON-RPC line that receive() should return.
            await asyncio.sleep(0.05)
            reader.feed_data(b"\n")
            await asyncio.sleep(0.05)
            reader.feed_data(b'{"jsonrpc": "2.0", "id": 1, "result": {"ok": true}}\n')
            reader.feed_eof()

        feeder = asyncio.create_task(feed_valid_later())
        try:
            msg = await asyncio.wait_for(transport.receive(), timeout=5)
        finally:
            await feeder
        assert msg == {"jsonrpc": "2.0", "id": 1, "result": {"ok": True}}


class TestTransportProtocolFiltering:
    """Transports skip JSON-encoded log scalars and wait for JSON-RPC objects."""

    @pytest.mark.asyncio
    async def test_stdio_transport_skips_json_scalars(self) -> None:
        """Guards PR #236 against treating JSON scalars as ACP responses."""
        reader = asyncio.StreamReader()
        reader.feed_data(b'"debug string from agent"\n')
        reader.feed_data(b'["debug", "list"]\n')
        reader.feed_data(b"123\n")
        reader.feed_data(b'{"jsonrpc": "2.0", "id": 1, "result": {"ok": true}}\n')
        reader.feed_eof()

        transport = StdioTransport(sys.executable, [])
        fake_process = MagicMock()
        fake_process.stdout = reader
        fake_process.stdin = MagicMock()
        transport._process = fake_process

        msg = await asyncio.wait_for(transport.receive(), timeout=5)
        assert msg == {"jsonrpc": "2.0", "id": 1, "result": {"ok": True}}

    @pytest.mark.asyncio
    async def test_container_transport_skips_json_scalars(self, tmp_path) -> None:
        """Guards PR #236 against treating JSON scalars as ACP responses."""
        fake_process = AsyncMock()
        fake_process.readline = AsyncMock(
            side_effect=[
                b'"debug string from agent"\n',
                b'["debug", "list"]\n',
                b'{"jsonrpc": "2.0", "id": 2, "result": {"ok": true}}\n',
            ]
        )
        fake_process.read_stderr = AsyncMock(return_value=b"")
        agent_log = tmp_path / "agent.log"
        transport = ContainerTransport(
            container_process=fake_process,
            command="agent acp",
            agent_log_path=agent_log,
        )

        await transport.start()
        try:
            msg = await asyncio.wait_for(transport.receive(), timeout=5)
        finally:
            await transport.close()

        assert msg == {"jsonrpc": "2.0", "id": 2, "result": {"ok": True}}
        log_text = agent_log.read_text()
        assert '"debug string from agent"' in log_text
        assert '["debug", "list"]' in log_text

    @pytest.mark.asyncio
    async def test_stdio_transport_skips_structured_json_logs(self) -> None:
        """Guards PR #236 against treating JSON object logs as ACP responses."""
        reader = asyncio.StreamReader()
        reader.feed_data(b'{"id": 100001, "level": "info", "message": "startup"}\n')
        reader.feed_data(b'{"jsonrpc": "2.0", "id": 100001, "result": {"ok": true}}\n')
        reader.feed_eof()

        transport = StdioTransport(sys.executable, [])
        fake_process = MagicMock()
        fake_process.stdout = reader
        fake_process.stdin = MagicMock()
        transport._process = fake_process
        client = ACPClient(transport)

        result = await asyncio.wait_for(client._read_until_response(100001), timeout=5)
        assert result == {"ok": True}

    @pytest.mark.asyncio
    async def test_container_transport_logs_structured_json_logs(
        self, tmp_path
    ) -> None:
        """Guards PR #236 against treating JSON object logs as ACP responses."""
        fake_process = AsyncMock()
        fake_process.readline = AsyncMock(
            side_effect=[
                b'{"id": 2, "level": "info", "message": "startup"}\n',
                b'{"jsonrpc": "2.0", "id": 2, "result": {"ok": true}}\n',
            ]
        )
        fake_process.read_stderr = AsyncMock(return_value=b"")
        agent_log = tmp_path / "agent.log"
        transport = ContainerTransport(
            container_process=fake_process,
            command="agent acp",
            agent_log_path=agent_log,
        )

        await transport.start()
        try:
            msg = await asyncio.wait_for(transport.receive(), timeout=5)
        finally:
            await transport.close()

        assert msg == {"jsonrpc": "2.0", "id": 2, "result": {"ok": True}}
        assert '{"id": 2, "level": "info", "message": "startup"}' in (
            agent_log.read_text()
        )


class TestContainerTransportStderrCapture:
    """Container agent stderr is drained concurrently and persisted separately."""

    @pytest.mark.asyncio
    async def test_large_stderr_does_not_block_json_rpc(self, tmp_path) -> None:
        """Guards the stderr-drain fix after commit 3572437 against pipe deadlock."""
        from benchflow.sandbox.process import LiveProcess

        stderr_size = 1024 * 1024
        stderr_tail = b"\nprovider-status=429\n"
        script = (
            "import sys\n"
            f"sys.stderr.buffer.write(b'x' * {stderr_size} + {stderr_tail!r})\n"
            "sys.stderr.buffer.flush()\n"
            'sys.stdout.write(\'{"jsonrpc":"2.0","id":7,"result":{"ok":true}}\\n\')\n'
            "sys.stdout.flush()\n"
            "sys.stdin.buffer.read(1)\n"
        )

        class LocalProcess(LiveProcess):
            async def start(self, command, env=None, cwd=None) -> None:
                self._reset_stderr_tail()
                self._process = await asyncio.create_subprocess_exec(
                    sys.executable,
                    "-c",
                    script,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )

        agent_log = tmp_path / "codex_acp.txt"
        transport = ContainerTransport(
            container_process=LocalProcess(),
            command="ignored-by-local-test-process",
            agent_log_path=agent_log,
        )

        await transport.start()
        try:
            message = await asyncio.wait_for(transport.receive(), timeout=5)
        finally:
            await transport.close()

        assert message == {"jsonrpc": "2.0", "id": 7, "result": {"ok": True}}
        assert (tmp_path / "codex_acp.stderr.txt").read_bytes() == (
            b"x" * stderr_size + stderr_tail
        )

    @pytest.mark.asyncio
    async def test_close_waits_for_stderr_tail(self, tmp_path) -> None:
        """Guards the stderr-drain fix after commit 3572437 against tail loss."""

        class ClosingTailProcess:
            def __init__(self) -> None:
                self.read_count = 0
                self.waiting_for_close = asyncio.Event()
                self.close_started = asyncio.Event()

            async def start(self, command, env=None, cwd=None) -> None:
                return None

            async def read_stderr(self) -> bytes:
                self.read_count += 1
                if self.read_count == 1:
                    return b"before-close\n"
                if self.read_count == 2:
                    self.waiting_for_close.set()
                    await self.close_started.wait()
                    return b"tail-without-newline"
                return b""

            async def close(self) -> None:
                self.close_started.set()

        process = ClosingTailProcess()
        agent_log = tmp_path / "agent.log"
        transport = ContainerTransport(
            container_process=process,
            command="agent acp",
            agent_log_path=agent_log,
        )

        await transport.start()
        await asyncio.wait_for(process.waiting_for_close.wait(), timeout=5)
        assert (tmp_path / "agent.stderr.log").read_bytes() == b"before-close\n"

        await asyncio.wait_for(transport.close(), timeout=5)

        assert (tmp_path / "agent.stderr.log").read_bytes() == (
            b"before-close\ntail-without-newline"
        )

    @pytest.mark.asyncio
    async def test_close_failure_cancels_stderr_drain(self, tmp_path) -> None:
        """Guards the stderr-drain fix after commit 3572437 against close hangs."""

        class FailingCloseProcess:
            def __init__(self) -> None:
                self.drain_started = asyncio.Event()
                self.drain_cancelled = asyncio.Event()

            async def start(self, command, env=None, cwd=None) -> None:
                return None

            async def read_stderr(self) -> bytes:
                self.drain_started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    self.drain_cancelled.set()
                    raise
                return b""

            async def close(self) -> None:
                raise RuntimeError("close failed")

        process = FailingCloseProcess()
        transport = ContainerTransport(
            container_process=process,
            command="agent acp",
            agent_log_path=tmp_path / "agent.log",
        )

        await transport.start()
        await asyncio.wait_for(process.drain_started.wait(), timeout=5)

        with pytest.raises(RuntimeError, match="close failed"):
            await asyncio.wait_for(transport.close(), timeout=5)

        assert process.drain_cancelled.is_set()
        assert transport._stderr_task is None
        assert transport._agent_log_file is None
        assert transport._agent_stderr_log_file is None

    @pytest.mark.asyncio
    async def test_close_bounds_stderr_drain_when_descendant_keeps_pipe_open(
        self, tmp_path, monkeypatch
    ) -> None:
        """A compose-provider descendant must not block transport close forever."""
        import benchflow.acp.container_transport as transport_module

        class HangingDrainProcess:
            def __init__(self) -> None:
                self.drain_started = asyncio.Event()
                self.drain_cancelled = asyncio.Event()

            async def start(self, command, env=None, cwd=None) -> None:
                return None

            async def read_stderr(self) -> bytes:
                self.drain_started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    self.drain_cancelled.set()
                    raise
                return b""

            async def close(self) -> None:
                return None

        monkeypatch.setattr(
            transport_module, "_STDERR_DRAIN_CLOSE_TIMEOUT_SEC", 0.01
        )
        process = HangingDrainProcess()
        transport = ContainerTransport(
            container_process=process,
            command="agent acp",
            agent_log_path=tmp_path / "agent.log",
        )

        await transport.start()
        await asyncio.wait_for(process.drain_started.wait(), timeout=5)
        await asyncio.wait_for(transport.close(), timeout=1)

        assert process.drain_cancelled.is_set()
        assert transport._stderr_task is None
        assert transport._agent_log_file is None
        assert transport._agent_stderr_log_file is None


class TestACPInterleaving:
    """Test that _read_until_response handles interleaved notifications and agent requests."""

    @pytest.mark.asyncio
    async def test_prompt_with_interleaved_notifications_and_request(self) -> None:
        """Notification, agent request, more notifications, then final response."""
        client = ACPClient(StdioTransport(sys.executable, [MOCK_AGENT_INTERLEAVED]))
        try:
            await client.connect()
            await client.initialize()
            await client.session_new()

            result = await client.prompt("Go!")
            assert result.stop_reason == StopReason.END_TURN

            session = client.session
            assert session is not None
            # Notifications processed: tool_call + tool_call_update + message chunk
            assert len(session.tool_calls) == 1
            assert session.tool_calls[0].status == ToolCallStatus.COMPLETED
            assert session.full_message == "done"
        finally:
            await client.close()


class TestConnectAcpModelSelection:
    """Verify connect_acp passes the right model string to set_model."""

    @staticmethod
    def _make_mocks():
        mock_session = MagicMock()
        mock_session.session_id = "s1"
        mock_init = MagicMock()
        mock_init.agent_info = None

        mock_acp = AsyncMock(spec=ACPClient)
        mock_acp.connect = AsyncMock()
        mock_acp.initialize = AsyncMock(return_value=mock_init)
        mock_acp.session_new = AsyncMock(return_value=mock_session)
        mock_acp.set_model = AsyncMock()
        mock_acp.close = AsyncMock()
        return mock_acp

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "model_in, expected_model",
        [
            # Registered vllm/ prefix stripped; HF org/model intact — this is
            # what pi-acp and other ACP agents need for downstream routing.
            ("vllm/Qwen/Qwen3.5-35B-A3B", "Qwen/Qwen3.5-35B-A3B"),
            ("zai/glm-5", "glm-5"),
            # Bare HF ID (no registered prefix) passes through unchanged.
            ("Qwen/Qwen3-Coder", "Qwen/Qwen3-Coder"),
            # Vertex ADC provider — prefix stripped like any other registered one.
            ("anthropic-vertex/claude-sonnet-4-6", "claude-sonnet-4-6"),
            # No prefix at all — unchanged.
            ("claude-sonnet-4-6", "claude-sonnet-4-6"),
        ],
        ids=["vllm-hf", "zai", "bare-hf", "vertex", "no-prefix"],
    )
    async def test_model_id_selection(self, model_in, expected_model, tmp_path):
        from benchflow.acp.runtime import connect_acp

        mock_acp = self._make_mocks()

        mock_env = AsyncMock()
        with (
            patch(
                "benchflow.acp.runtime.DockerProcess.from_sandbox_env",
                return_value=MagicMock(),
            ),
            patch("benchflow.acp.runtime.ContainerTransport", return_value=MagicMock()),
            patch("benchflow.acp.runtime.ACPClient", return_value=mock_acp),
        ):
            await connect_acp(
                env=mock_env,
                agent="test-agent",
                agent_launch="test-agent",
                agent_env={},
                sandbox_user=None,
                model=model_in,
                rollout_dir=tmp_path,
                environment="docker",
                agent_cwd="/app",
            )

        mock_acp.set_model.assert_awaited_once_with(expected_model)

    @pytest.mark.asyncio
    async def test_openhands_skips_set_model(self, tmp_path):
        from benchflow.acp.runtime import connect_acp

        mock_acp = self._make_mocks()
        mock_env = AsyncMock()
        with (
            patch(
                "benchflow.acp.runtime.DockerProcess.from_sandbox_env",
                return_value=MagicMock(),
            ),
            patch("benchflow.acp.runtime.ContainerTransport", return_value=MagicMock()),
            patch("benchflow.acp.runtime.ACPClient", return_value=mock_acp),
        ):
            await connect_acp(
                env=mock_env,
                agent="openhands",
                agent_launch="openhands acp --always-approve --override-with-envs",
                agent_env={},
                sandbox_user=None,
                model="gemini-3.1-flash-lite-preview",
                rollout_dir=tmp_path,
                environment="docker",
                agent_cwd="/app",
            )

        mock_acp.set_model.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_claude_bedrock_sets_model_from_provider_mapping(self, tmp_path):
        from benchflow.acp.runtime import connect_acp

        mock_acp = self._make_mocks()
        mock_env = AsyncMock()
        with (
            patch(
                "benchflow.acp.runtime.DockerProcess.from_sandbox_env",
                return_value=MagicMock(),
            ),
            patch("benchflow.acp.runtime.ContainerTransport", return_value=MagicMock()),
            patch("benchflow.acp.runtime.ACPClient", return_value=mock_acp),
        ):
            await connect_acp(
                env=mock_env,
                agent="claude-agent-acp",
                agent_launch="claude-agent-acp",
                agent_env={
                    "CLAUDE_CODE_USE_BEDROCK": "1",
                    "ANTHROPIC_MODEL": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
                },
                sandbox_user=None,
                model="aws-bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
                rollout_dir=tmp_path,
                environment="docker",
                agent_cwd="/app",
            )

        mock_acp.set_model.assert_awaited_once_with(
            "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
        )

    @pytest.mark.asyncio
    async def test_daytona_dind_uses_pty_transport(self, tmp_path):
        """Daytona compose tasks use PTY transport to avoid SSH pipe-closed failures."""
        from benchflow.acp.runtime import connect_acp

        mock_acp = self._make_mocks()
        mock_env = MagicMock()
        mock_env.exec = AsyncMock(return_value=MagicMock(return_code=1, stdout=""))
        mock_env._strategy = MagicMock()
        mock_env._strategy._compose_cmd = MagicMock(return_value="docker compose -p t")

        with (
            patch(
                "benchflow.acp.runtime.DaytonaPtyProcess.from_sandbox_env",
                new_callable=AsyncMock,
                return_value=MagicMock(),
            ) as mock_pty,
            patch(
                "benchflow.acp.runtime.DaytonaProcess.from_sandbox_env",
                new_callable=AsyncMock,
                return_value=MagicMock(),
            ) as mock_ssh,
            patch("benchflow.acp.runtime.ContainerTransport", return_value=MagicMock()),
            patch("benchflow.acp.runtime.ACPClient", return_value=mock_acp),
        ):
            await connect_acp(
                env=mock_env,
                agent="test-agent",
                agent_launch="test-agent",
                agent_env={},
                sandbox_user=None,
                model=None,
                rollout_dir=tmp_path,
                environment="daytona",
                agent_cwd="/app",
            )

        mock_pty.assert_awaited_once_with(mock_env)
        mock_ssh.assert_not_awaited()
