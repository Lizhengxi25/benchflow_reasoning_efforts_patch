from __future__ import annotations

import http.client
import json
import stat
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import httpx
import pytest

from benchflow.providers import litellm_runtime as runtime_mod
from benchflow.providers.litellm_config import LiteLLMRoute
from benchflow.providers.litellm_runtime import _start_host_litellm
from benchflow.providers.runtime import ensure_litellm_runtime
from benchflow.providers.wire_capture import (
    OMIT_REASONING,
    PROVIDER_REQUEST_FILTER_ENV,
    ProviderWireProxy,
    filter_request_payload,
)


def _stop_server(server: ThreadingHTTPServer, thread: threading.Thread) -> None:
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


def test_omit_reasoning_strips_every_supported_control_at_provider_boundary():
    payload = {
        "messages": [{"role": "user", "content": "hi"}],
        "include_reasoning": True,
        "reasoning": {"effort": "high"},
        "reasoning_effort": "high",
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": "high", "format": {"type": "json"}},
        "include": ["reasoning.encrypted_content", "message.output_text.logprobs"],
        "extra_body": {
            "reasoning": {"effort": "medium"},
            "output_config": {"effort": "medium"},
            "route": "fallback",
        },
        "litellm_extra_body": {"reasoning_effort": "low"},
    }

    filtered = filter_request_payload(
        payload,
        OMIT_REASONING,
        protocol="openai-completions",
    )

    assert filtered == {
        "messages": [{"role": "user", "content": "hi"}],
        "output_config": {"format": {"type": "json"}},
        "include": ["message.output_text.logprobs"],
        "extra_body": {"route": "fallback"},
    }
    assert payload["reasoning"] == {"effort": "high"}


def test_explicit_effort_uses_openrouter_reasoning_shape_for_chat_completions():
    filtered = filter_request_payload(
        {
            "messages": [{"role": "user", "content": "hi"}],
            "reasoning_effort": "medium",
            "thinking": {"type": "adaptive"},
        },
        "reasoning-effort:xhigh",
        protocol="openai-completions",
    )

    assert filtered == {
        "messages": [{"role": "user", "content": "hi"}],
        "reasoning": {"effort": "xhigh"},
    }


def test_wire_proxy_captures_exact_bodies_raw_sse_and_sanitized_metadata(
    tmp_path: Path,
):
    observed: dict[str, Any] = {}
    response_body = (
        b'data: {"type":"response.output_text.delta","delta":"ok"}\n\ndata: [DONE]\n\n'
    )

    class UpstreamHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers["content-length"])
            observed["path"] = self.path
            observed["body"] = self.rfile.read(length)
            observed["authorization"] = self.headers["authorization"]
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(response_body)))
            self.send_header("Set-Cookie", "provider-session=secret")
            self.end_headers()
            self.wfile.write(response_body)

        def log_message(self, format: str, *args: Any) -> None:
            return

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), UpstreamHandler)
    upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()
    capture_root = tmp_path / "agent" / "model_io"
    proxy = ProviderWireProxy(
        upstream=f"http://127.0.0.1:{upstream.server_port}/api/v1",
        mode=OMIT_REASONING,
        capture_dir=capture_root,
    )
    logical_body = (
        b'{"model":"demo","messages":[{"role":"user","content":"hi"}],'
        b'"stream":true,"reasoning_effort":"high",'
        b'"reasoning":{"effort":"high"},"thinking":{"type":"adaptive"}}'
    )
    api_key = "provider-secret-value"

    try:
        endpoint = httpx.URL(proxy.base_url)
        connection = http.client.HTTPConnection(endpoint.host, endpoint.port, timeout=5)
        connection.request(
            "POST",
            "/chat/completions?api_key=query-secret",
            body=logical_body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        response = connection.getresponse()
        assert response.status == 200
        assert response.read() == response_body
        connection.close()
    finally:
        proxy.stop()
        _stop_server(upstream, upstream_thread)

    assert observed["path"] == "/api/v1/chat/completions?api_key=query-secret"
    assert observed["authorization"] == f"Bearer {api_key}"
    provider_body = observed["body"]
    assert json.loads(provider_body) == {
        "model": "demo",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    }

    request_dir = capture_root / "000001"
    assert (request_dir / "logical_request.body").read_bytes() == logical_body
    assert (
        json.loads((request_dir / "logical_request.json").read_text())[
            "reasoning_effort"
        ]
        == "high"
    )
    assert (request_dir / "provider_request.body").read_bytes() == provider_body
    assert json.loads((request_dir / "provider_request.json").read_text()) == (
        json.loads(provider_body)
    )
    assert (request_dir / "provider_response.sse").read_bytes() == response_body

    metadata = json.loads((request_dir / "metadata.json").read_text())
    assert metadata["filter_mode"] == OMIT_REASONING
    assert metadata["protocol"] == "openai-completions"
    assert metadata["transport_complete"] is True
    assert metadata["error"] is None
    assert metadata["request"]["logical_headers"]["Authorization"] == "<redacted>"
    assert metadata["request"]["provider_headers"]["Authorization"] == "<redacted>"
    assert "query-secret" not in metadata["request"]["url"]
    assert metadata["response"]["headers"]["Set-Cookie"] == "<redacted>"
    assert metadata["response"]["status"] == 200
    assert metadata["response"]["body_bytes"] == len(response_body)
    assert metadata["timing"]["time_to_headers_ms"] is not None
    assert metadata["timing"]["time_to_first_byte_ms"] is not None
    assert metadata["timing"]["duration_ms"] is not None
    assert stat.S_IMODE(capture_root.stat().st_mode) == 0o700
    for artifact in request_dir.iterdir():
        assert stat.S_IMODE(artifact.stat().st_mode) == 0o600


def test_wire_proxy_keeps_partial_response_and_marks_incomplete(tmp_path: Path):
    partial = b"data: partial\n\n"

    class UpstreamHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers["content-length"])
            self.rfile.read(length)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(partial) + 50))
            self.end_headers()
            self.wfile.write(partial)
            self.wfile.flush()
            self.close_connection = True

        def log_message(self, format: str, *args: Any) -> None:
            return

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), UpstreamHandler)
    upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()
    capture_root = tmp_path / "model_io"
    proxy = ProviderWireProxy(
        upstream=f"http://127.0.0.1:{upstream.server_port}",
        mode=OMIT_REASONING,
        capture_dir=capture_root,
    )
    try:
        with httpx.Client(timeout=5) as client:
            response = client.post(
                f"{proxy.base_url}/chat/completions",
                json={"model": "demo", "messages": [], "stream": True},
            )
            assert response.status_code == 200
            assert response.content == partial
    finally:
        proxy.stop()
        _stop_server(upstream, upstream_thread)

    request_dir = capture_root / "000001"
    assert (request_dir / "provider_response.sse").read_bytes() == partial
    metadata = json.loads((request_dir / "metadata.json").read_text())
    assert metadata["transport_complete"] is False
    assert "IncompleteRead" in metadata["error"]
    assert metadata["response"]["body_bytes"] == len(partial)


def test_filter_only_proxy_does_not_create_raw_body_artifacts(tmp_path: Path):
    """Raw model I/O stays opt-in even when final-boundary filtering is active."""
    observed: dict[str, Any] = {}
    response_body = b'{"id":"response-1","choices":[]}'

    class UpstreamHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers["content-length"])
            observed["body"] = self.rfile.read(length)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response_body)))
            self.end_headers()
            self.wfile.write(response_body)

        def log_message(self, format: str, *args: Any) -> None:
            return

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), UpstreamHandler)
    upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()
    proxy = ProviderWireProxy(
        upstream=f"http://127.0.0.1:{upstream.server_port}",
        mode=OMIT_REASONING,
    )
    try:
        response = httpx.post(
            f"{proxy.base_url}/chat/completions",
            json={
                "model": "demo",
                "messages": [],
                "reasoning": {"effort": "high"},
            },
            timeout=5,
        )
        assert response.status_code == 200
    finally:
        proxy.stop()
        _stop_server(upstream, upstream_thread)

    assert json.loads(observed["body"]) == {
        "model": "demo",
        "messages": [],
    }
    assert list(tmp_path.iterdir()) == []


class _FakeLiteLLMServer:
    def __init__(self, route: LiteLLMRoute):
        self.route = route
        self.base_url = "http://127.0.0.1:45678"
        self.stopped = False

    async def is_running(self) -> bool:
        return not self.stopped

    async def stop(self) -> None:
        self.stopped = True

    def start_live_capture(self, path: Path) -> None:
        return


@pytest.mark.asyncio
async def test_runtime_does_not_enable_raw_capture_from_live_trajectory_alone(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    """The always-on logical trajectory must not implicitly enable raw bodies."""
    starts: list[dict[str, Any]] = []

    async def fake_start(**kwargs: Any) -> _FakeLiteLLMServer:
        starts.append(kwargs)
        return _FakeLiteLLMServer(kwargs["route"])

    monkeypatch.setattr(runtime_mod, "_start_host_litellm", fake_start)
    trajectory_path = tmp_path / "run" / "trajectory" / "llm_trajectory.jsonl"

    _updated, runtime = await ensure_litellm_runtime(
        agent="codex-acp",
        agent_env={"OPENAI_API_KEY": "sk-test"},
        model="openai/gpt-4.1-mini",
        runtime=None,
        environment="docker",
        live_trajectory_path=trajectory_path,
    )

    assert starts[0]["wire_capture_dir"] is None
    assert not (tmp_path / "run" / "agent" / "model_io").exists()
    assert runtime is not None
    await runtime.server.stop()


@pytest.mark.asyncio
async def test_runtime_opt_in_targets_rollout_agent_model_io(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    """The explicit capture bit reaches the host LiteLLM/provider boundary."""
    starts: list[dict[str, Any]] = []

    async def fake_start(**kwargs: Any) -> _FakeLiteLLMServer:
        starts.append(kwargs)
        return _FakeLiteLLMServer(kwargs["route"])

    monkeypatch.setattr(runtime_mod, "_start_host_litellm", fake_start)
    trajectory_path = tmp_path / "run" / "trajectory" / "llm_trajectory.jsonl"

    _updated, runtime = await ensure_litellm_runtime(
        agent="codex-acp",
        agent_env={"OPENAI_API_KEY": "sk-test"},
        model="openai/gpt-4.1-mini",
        runtime=None,
        environment="docker",
        live_trajectory_path=trajectory_path,
        capture_model_io=True,
    )

    assert starts[0]["wire_capture_dir"] == (tmp_path / "run" / "agent" / "model_io")
    assert runtime is not None
    await runtime.server.stop()


@pytest.mark.asyncio
async def test_runtime_restarts_when_final_boundary_filter_changes(
    monkeypatch: pytest.MonkeyPatch,
):
    """Runtime reuse must include final-boundary filter policy in its key."""
    starts: list[dict[str, Any]] = []

    async def fake_start(**kwargs: Any) -> _FakeLiteLLMServer:
        starts.append(kwargs)
        return _FakeLiteLLMServer(kwargs["route"])

    monkeypatch.setattr(runtime_mod, "_start_host_litellm", fake_start)
    _updated, first = await ensure_litellm_runtime(
        agent="codex-acp",
        agent_env={
            "OPENAI_API_KEY": "sk-test",
            PROVIDER_REQUEST_FILTER_ENV: OMIT_REASONING,
        },
        model="openai/gpt-4.1-mini",
        runtime=None,
        environment="docker",
    )
    assert first is not None
    first_server = first.server

    _updated, second = await ensure_litellm_runtime(
        agent="codex-acp",
        agent_env={
            "OPENAI_API_KEY": "sk-test",
            PROVIDER_REQUEST_FILTER_ENV: "reasoning-effort:low",
        },
        model="openai/gpt-4.1-mini",
        runtime=first,
        environment="docker",
    )

    assert first_server.stopped is True
    assert second is not first
    assert len(starts) == 2
    assert starts[0]["wire_capture_dir"] is None
    assert starts[1]["wire_capture_dir"] is None
    assert second is not None
    await second.server.stop()


@pytest.mark.asyncio
async def test_capture_fails_fast_for_sandbox_local_provider_runtime(tmp_path: Path):
    """Unsupported capture placement must never silently produce no artifacts."""
    with pytest.raises(RuntimeError, match="host/Docker LiteLLM only"):
        await ensure_litellm_runtime(
            agent="openhands",
            agent_env={"OPENAI_API_KEY": "sk-test"},
            model="openai/gpt-4.1-mini",
            runtime=None,
            environment="daytona",
            sandbox=object(),
            live_trajectory_path=(
                tmp_path / "run" / "trajectory" / "llm_trajectory.jsonl"
            ),
            capture_model_io=True,
        )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_host_litellm_start_path_captures_fake_provider_wire(
    tmp_path: Path,
):
    observed: dict[str, Any] = {}
    chunks = [
        {
            "id": "chatcmpl-wire-test",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "wire-test",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "ok"},
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl-wire-test",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "wire-test",
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
        },
    ]
    response_body = (
        b"".join(
            f"data: {json.dumps(chunk, separators=(',', ':'))}\n\n".encode()
            for chunk in chunks
        )
        + b"data: [DONE]\n\n"
    )

    class FakeProviderHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers["content-length"])
            observed["path"] = self.path
            observed["body"] = self.rfile.read(length)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(response_body)))
            self.end_headers()
            self.wfile.write(response_body)

        def log_message(self, format: str, *args: Any) -> None:
            return

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), FakeProviderHandler)
    upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()
    route = LiteLLMRoute(
        requested_model="openrouter/test/wire-model",
        model_alias="benchflow-wire-model",
        upstream_model="openai/wire-model",
        provider_name="openrouter",
        litellm_params={
            "model": "openai/wire-model",
            "api_base": f"http://127.0.0.1:{upstream.server_port}/api/v1",
            "api_key": "fake-provider-key",
        },
    )
    master_key = "sk-benchflow-wire-test"
    capture_root = tmp_path / "agent" / "model_io"
    runner = None
    try:
        runner = await _start_host_litellm(
            route=route,
            master_key=master_key,
            agent_env={PROVIDER_REQUEST_FILTER_ENV: OMIT_REASONING},
            environment="local",
            session_id="wire-test",
            agent_name="openhands",
            wire_capture_dir=capture_root,
        )
        async with (
            httpx.AsyncClient(timeout=30) as client,
            client.stream(
                "POST",
                f"{runner.base_url}/v1/chat/completions",
                headers={"Authorization": f"Bearer {master_key}"},
                json={
                    "model": route.model_alias,
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": True,
                    "reasoning": {"effort": "high"},
                    "reasoning_effort": "high",
                },
            ) as response,
        ):
            assert response.status_code == 200
            downstream = b"".join([chunk async for chunk in response.aiter_bytes()])
        assert b"ok" in downstream
    finally:
        if runner is not None:
            await runner.stop()
        _stop_server(upstream, upstream_thread)

    assert observed["path"] == "/api/v1/chat/completions"
    final_body = json.loads(observed["body"])
    assert "reasoning" not in final_body
    assert "reasoning_effort" not in final_body
    request_dir = capture_root / "000001"
    assert (request_dir / "logical_request.body").is_file()
    assert (request_dir / "provider_request.body").read_bytes() == observed["body"]
    assert (request_dir / "provider_response.sse").read_bytes() == response_body
    metadata = json.loads((request_dir / "metadata.json").read_text())
    assert metadata["transport_complete"] is True
    assert metadata["response"]["status"] == 200
    debug_dir = capture_root.parent / "litellm" / "wire-test"
    for filename in ("stdout.log", "stderr.log", "callback.jsonl"):
        path = debug_dir / filename
        assert path.is_file()
        assert path.stat().st_mode & 0o777 == 0o600
