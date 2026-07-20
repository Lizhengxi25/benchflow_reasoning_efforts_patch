"""Request filtering used to preserve true vendor-default reasoning semantics."""

import http.client
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

import pytest

from benchflow.agents.registry import _request_filtering_js_launcher
from benchflow.agents.request_filter_proxy import (
    OMIT_REASONING,
    RequestFilterHandler,
    RequestFilterHTTPServer,
    filter_request_payload,
    reasoning_effort_filter,
    request_protocol,
    upstream_request_path,
    validate_filter_mode,
)
from benchflow.rollout import _apply_reasoning_effort


def test_generated_request_filter_launchers_compile():
    """Guards the Qwen3-Coder-Next pioneer commit's installed launch scripts."""
    for agent in ("claude-agent-acp", "codex-acp"):
        compile(_request_filtering_js_launcher(agent), f"<{agent}>", "exec")


def test_claude_code_explicit_effort_uses_native_cli_flag():
    """Guards the Qwen3-Coder-Next pioneer commit's extensible Claude path."""
    assert (
        _apply_reasoning_effort("claude-agent-acp", "claude-agent-acp", "high")
        == "claude-agent-acp --effort high"
    )


def test_omit_reasoning_filters_claude_code_controls_without_mutating_input():
    """Guards the Qwen3-Coder-Next pioneer commit's Claude request boundary."""
    payload = {
        "model": "qwen/qwen3-coder-next",
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": "high", "format": {"type": "json"}},
        "messages": [],
    }

    filtered = filter_request_payload(payload, OMIT_REASONING)

    assert "thinking" not in filtered
    assert filtered["output_config"] == {"format": {"type": "json"}}
    assert payload["thinking"] == {"type": "adaptive"}
    assert payload["output_config"]["effort"] == "high"


def test_omit_reasoning_filters_codex_responses_controls():
    """Guards the Qwen3-Coder-Next pioneer commit's Codex request boundary."""
    filtered = filter_request_payload(
        {
            "model": "qwen/qwen3-coder-next",
            "reasoning": {"effort": "medium", "summary": "auto"},
            "reasoning_effort": "medium",
            "include_reasoning": True,
            "include": ["reasoning.encrypted_content", "message.output_text.logprobs"],
            "input": [],
        },
        OMIT_REASONING,
    )

    assert "reasoning" not in filtered
    assert "reasoning_effort" not in filtered
    assert "include_reasoning" not in filtered
    assert filtered["include"] == ["message.output_text.logprobs"]


def test_explicit_effort_filter_modes_are_strictly_validated():
    assert reasoning_effort_filter("xhigh") == "reasoning-effort:xhigh"
    assert validate_filter_mode("reasoning-effort:none") == "reasoning-effort:none"

    for mode in ("reasoning-effort:", "reasoning-effort:default", "other"):
        with pytest.raises(ValueError, match="unsupported"):
            validate_filter_mode(mode)


def test_explicit_effort_normalizes_claude_anthropic_request():
    payload = {
        "model": "deepseek/deepseek-v4-pro",
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": "max", "format": {"type": "json"}},
        "messages": [],
    }

    filtered = filter_request_payload(
        payload,
        reasoning_effort_filter("xhigh"),
        protocol="anthropic-messages",
    )

    assert filtered["thinking"] == {"type": "adaptive"}
    assert filtered["output_config"] == {
        "format": {"type": "json"},
        "effort": "xhigh",
    }


def test_explicit_none_disables_claude_thinking():
    filtered = filter_request_payload(
        {
            "model": "tencent/hy3",
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "high"},
            "messages": [],
        },
        reasoning_effort_filter("none"),
        protocol="anthropic-messages",
    )

    assert filtered["thinking"] == {"type": "disabled"}
    assert "output_config" not in filtered


def test_explicit_effort_normalizes_codex_responses_request():
    filtered = filter_request_payload(
        {
            "model": "deepseek/deepseek-v4-flash",
            "reasoning": {"effort": "medium", "summary": "auto"},
            "input": [],
        },
        reasoning_effort_filter("high"),
        protocol="openai-responses",
    )

    assert filtered["reasoning"] == {"effort": "high"}


def test_request_protocol_only_matches_generation_endpoints():
    assert request_protocol("/responses?beta=1") == "openai-responses"
    assert request_protocol("/v1/messages") == "anthropic-messages"
    assert request_protocol("/v1/messages/count_tokens") is None
    assert request_protocol("/_benchflow/request-filter-health") is None


def test_upstream_request_path_preserves_base_path_and_query():
    """Guards the Qwen3-Coder-Next pioneer commit's OpenRouter URL routing."""
    assert (
        upstream_request_path(
            urlsplit("https://openrouter.ai/api/v1"),
            "/responses?beta=1",
        )
        == "/api/v1/responses?beta=1"
    )
    assert (
        upstream_request_path(
            urlsplit("https://openrouter.ai/api"),
            "/v1/messages",
        )
        == "/api/v1/messages"
    )


def test_loopback_proxy_filters_and_forwards_response_bytes():
    """Guards the Qwen3-Coder-Next pioneer commit's live proxy transport."""
    captured = {}
    response_body = b"event: message_stop\ndata: {}\n\n"

    class UpstreamHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            captured["path"] = self.path
            length = int(self.headers["content-length"])
            captured["body"] = json.loads(self.rfile.read(length))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(response_body)))
            self.end_headers()
            self.wfile.write(response_body)

        def log_message(self, _format, *_args):
            return

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), UpstreamHandler)
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    proxy = RequestFilterHTTPServer(
        ("127.0.0.1", 0),
        RequestFilterHandler,
        upstream=f"http://127.0.0.1:{upstream.server_port}/api",
        mode=OMIT_REASONING,
    )
    threading.Thread(target=proxy.serve_forever, daemon=True).start()

    try:
        connection = http.client.HTTPConnection(
            "127.0.0.1", proxy.server_port, timeout=5
        )
        connection.request(
            "POST",
            "/v1/messages",
            body=json.dumps(
                {
                    "model": "qwen/qwen3-coder-next",
                    "thinking": {"type": "adaptive"},
                    "output_config": {"effort": "high"},
                    "messages": [],
                }
            ),
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        assert response.status == 200
        assert response.read() == response_body
        connection.close()
    finally:
        proxy.shutdown()
        proxy.server_close()
        upstream.shutdown()
        upstream.server_close()

    assert captured["path"] == "/api/v1/messages"
    assert captured["body"] == {
        "model": "qwen/qwen3-coder-next",
        "messages": [],
    }


def test_loopback_proxy_forwards_normalized_xhigh_request():
    captured = {}

    class UpstreamHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers["content-length"])
            captured["body"] = json.loads(self.rfile.read(length))
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, _format, *_args):
            return

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), UpstreamHandler)
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    proxy = RequestFilterHTTPServer(
        ("127.0.0.1", 0),
        RequestFilterHandler,
        upstream=f"http://127.0.0.1:{upstream.server_port}/api",
        mode=reasoning_effort_filter("xhigh"),
    )
    threading.Thread(target=proxy.serve_forever, daemon=True).start()

    try:
        connection = http.client.HTTPConnection(
            "127.0.0.1", proxy.server_port, timeout=5
        )
        connection.request(
            "POST",
            "/v1/messages",
            body=json.dumps(
                {
                    "model": "deepseek/deepseek-v4-pro",
                    "thinking": {"type": "adaptive"},
                    "output_config": {"effort": "max"},
                    "messages": [],
                }
            ),
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        assert response.status == 200
        response.read()
        connection.close()
    finally:
        proxy.shutdown()
        proxy.server_close()
        upstream.shutdown()
        upstream.server_close()

    assert captured["body"] == {
        "model": "deepseek/deepseek-v4-pro",
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": "xhigh"},
        "messages": [],
    }
