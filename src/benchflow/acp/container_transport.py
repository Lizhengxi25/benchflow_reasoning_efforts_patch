"""ACP transport over a live stdio pipe to a sandbox process."""

import asyncio
import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, BinaryIO, TextIO

from benchflow.sandbox.process import LiveProcess
from benchflow.trajectories.types import (
    redact_trajectory_obj,
    redact_trajectory_text,
)

from .transport import Transport, decode_json_rpc_message

logger = logging.getLogger(__name__)


class ContainerTransport(Transport):
    """ACP transport that speaks to an agent running inside a sandbox.

    Uses a LiveProcess (DockerProcess or DaytonaProcess) to maintain a live
    stdin/stdout connection. Non-JSON lines from the agent (debug output,
    errors, warnings) are captured to a log file if agent_log_path is set.
    The process's separate stderr stream is drained concurrently into a sibling
    ``*.stderr.*`` file so a verbose agent cannot block on a full pipe.
    """

    def __init__(
        self,
        container_process: LiveProcess,
        command: str,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        agent_log_path: Path | None = None,
    ):
        self._cp = container_process
        self._command = command
        self._env = env or {}
        self._cwd = cwd
        self._agent_log_path = agent_log_path
        self._agent_log_file: TextIO | None = None
        self._agent_stderr_log_path = (
            agent_log_path.with_name(
                f"{agent_log_path.stem}.stderr{agent_log_path.suffix}"
            )
            if agent_log_path
            else None
        )
        self._agent_stderr_log_file: BinaryIO | None = None
        self._agent_wire_log_path = (
            agent_log_path.with_name(f"{agent_log_path.stem}.acp_wire.jsonl")
            if agent_log_path
            else None
        )
        self._agent_wire_log_file: TextIO | None = None
        self._agent_wire_log_error_reported = False
        self._stderr_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Start the agent process inside the sandbox."""
        try:
            if self._agent_log_path:
                self._agent_log_path.parent.mkdir(parents=True, exist_ok=True)
                # _connect_acp_session reuses these paths across retries. Keep
                # both logs lazy, but clear stale content before each attempt.
                self._agent_log_path.unlink(missing_ok=True)
                assert self._agent_stderr_log_path is not None
                self._agent_stderr_log_path.unlink(missing_ok=True)
                assert self._agent_wire_log_path is not None
                try:
                    self._agent_wire_log_path.unlink(missing_ok=True)
                except OSError:
                    logger.warning(
                        "Failed to reset ACP wire capture; capture is disabled for "
                        "this transport",
                        exc_info=True,
                    )
                    self._agent_wire_log_path = None
            await self._cp.start(
                command=self._command,
                env=self._env,
                cwd=self._cwd,
            )
        except BaseException:
            self._close_log_files()
            raise
        self._stderr_task = asyncio.create_task(
            self._drain_stderr(), name="benchflow-agent-stderr"
        )
        logger.info(f"ContainerTransport: agent started ({self._command})")

    async def _drain_stderr(self) -> None:
        """Drain stderr through EOF, persisting every byte when configured."""
        try:
            while True:
                chunk = await self._cp.read_stderr()
                if not chunk:
                    return
                if not isinstance(chunk, (bytes, bytearray)):
                    raise TypeError("LiveProcess.read_stderr() must return bytes")
                if self._agent_stderr_log_path:
                    if self._agent_stderr_log_file is None:
                        self._agent_stderr_log_file = self._agent_stderr_log_path.open(
                            "wb"
                        )
                    self._agent_stderr_log_file.write(chunk)
                    self._agent_stderr_log_file.flush()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning(
                "Failed while draining container agent stderr", exc_info=True
            )

    def _close_log_files(self) -> None:
        if self._agent_log_file:
            self._agent_log_file.close()
            self._agent_log_file = None
        if self._agent_stderr_log_file:
            self._agent_stderr_log_file.close()
            self._agent_stderr_log_file = None
        self._discard_wire_log_file()

    def _discard_wire_log_file(self) -> None:
        wire_log_file = self._agent_wire_log_file
        self._agent_wire_log_file = None
        if wire_log_file is None:
            return
        try:
            wire_log_file.close()
        except Exception:
            # A diagnostic sink must never replace an ACP/process error during
            # cleanup. The write path already reports its primary failure.
            logger.warning("Failed to close ACP wire capture", exc_info=True)

    def _open_wire_log_file(self) -> TextIO:
        assert self._agent_wire_log_path is not None
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW

        descriptor: int | None = os.open(self._agent_wire_log_path, flags, 0o600)
        try:
            os.fchmod(descriptor, 0o600)
            wire_log_file = os.fdopen(
                descriptor,
                "a",
                encoding="utf-8",
            )
            descriptor = None
            return wire_log_file
        finally:
            if descriptor is not None:
                os.close(descriptor)

    def _write_wire(
        self,
        *,
        direction: str,
        raw: str,
        message: dict[str, Any],
    ) -> None:
        """Persist one redacted JSON-RPC frame without changing transport data."""
        if self._agent_wire_log_path is None:
            return
        try:
            if self._agent_wire_log_file is None:
                self._agent_wire_log_file = self._open_wire_log_file()
            record = {
                "timestamp": datetime.now(UTC).isoformat(),
                "direction": direction,
                "raw": redact_trajectory_text(raw),
                "message": redact_trajectory_obj(message),
            }
            serialized = (
                json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
            written = self._agent_wire_log_file.write(serialized)
            if written != len(serialized):
                raise OSError(
                    f"short ACP wire capture write: {written}/{len(serialized)}"
                )
            self._agent_wire_log_file.flush()
        except Exception:
            self._discard_wire_log_file()
            if not self._agent_wire_log_error_reported:
                logger.warning(
                    "ACP wire capture failed; protocol traffic will continue and "
                    "capture will retry on the next frame",
                    exc_info=True,
                )
                self._agent_wire_log_error_reported = True
        else:
            if self._agent_wire_log_error_reported:
                logger.info("ACP wire capture resumed after a previous failure")
                self._agent_wire_log_error_reported = False

    async def send(self, message: dict[str, Any]) -> None:
        """Send a JSON-RPC message to the agent."""
        data = json.dumps(message)
        self._write_wire(
            direction="client_to_agent",
            raw=data,
            message=message,
        )
        await self._cp.writeline(data)

    async def receive(self) -> dict[str, Any]:
        """Receive a JSON-RPC message from the agent."""
        while True:
            line = await self._cp.readline()
            text = line.decode(errors="replace").strip()
            if not text:
                continue
            message = decode_json_rpc_message(text)
            if message is not None:
                self._write_wire(
                    direction="agent_to_client",
                    raw=text,
                    message=message,
                )
                return message
            # Capture non-protocol output (agent debug logs, errors, warnings).
            if self._agent_log_path:
                if self._agent_log_file is None:
                    self._agent_log_file = self._agent_log_path.open("w")
                self._agent_log_file.write(text + "\n")
                self._agent_log_file.flush()
            logger.debug(f"Non-JSON-RPC from container agent: {text[:200]}")

    async def close(self) -> None:
        """Terminate the agent, drain stderr through EOF, then close logs."""
        process_closed = False
        try:
            await self._cp.close()
            process_closed = True
        finally:
            if self._stderr_task:
                if not process_closed:
                    self._stderr_task.cancel()
                await asyncio.gather(self._stderr_task, return_exceptions=True)
                self._stderr_task = None
            self._close_log_files()
