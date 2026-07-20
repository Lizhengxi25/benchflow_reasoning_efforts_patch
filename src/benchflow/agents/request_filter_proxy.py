#!/usr/bin/env python3
"""Small loopback proxy for normalizing harness-added request controls.

The caller still selects the model and upstream provider normally.  This proxy
only applies an explicit, named body transformation before forwarding the
request byte stream to that provider.
"""

from __future__ import annotations

import argparse
import copy
import http.client
import json
import ssl
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import SplitResult, urlsplit

OMIT_REASONING = "omit-reasoning"
REASONING_EFFORT_PREFIX = "reasoning-effort:"
SUPPORTED_REASONING_EFFORTS = (
    "max",
    "xhigh",
    "high",
    "medium",
    "low",
    "minimal",
    "none",
)
HEALTH_PATH = "/_benchflow/request-filter-health"
_HOP_BY_HOP_HEADERS = {
    "connection",
    "content-length",
    "host",
    "proxy-connection",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}


def reasoning_effort_filter(effort: str) -> str:
    """Build a validated request-filter mode for an explicit effort."""
    if effort not in SUPPORTED_REASONING_EFFORTS:
        raise ValueError(f"unsupported reasoning effort {effort!r}")
    return f"{REASONING_EFFORT_PREFIX}{effort}"


def validate_filter_mode(mode: str) -> str:
    """Validate and return a request-filter mode."""
    if mode == OMIT_REASONING:
        return mode
    if mode.startswith(REASONING_EFFORT_PREFIX):
        reasoning_effort_filter(mode.removeprefix(REASONING_EFFORT_PREFIX))
        return mode
    raise ValueError(f"unsupported request filter {mode!r}")


def request_protocol(path: str) -> str | None:
    """Identify generation endpoints without matching token-count helpers."""
    normalized = path.partition("?")[0].rstrip("/")
    if normalized.endswith("/responses"):
        return "openai-responses"
    if normalized.endswith("/messages"):
        return "anthropic-messages"
    return None


def _strip_reasoning_controls(payload: dict) -> dict:
    filtered = copy.deepcopy(payload)
    for key in (
        "include_reasoning",
        "reasoning",
        "reasoning_effort",
        "thinking",
    ):
        filtered.pop(key, None)

    output_config = filtered.get("output_config")
    if isinstance(output_config, dict):
        output_config.pop("effort", None)
        if not output_config:
            filtered.pop("output_config", None)

    include = filtered.get("include")
    if isinstance(include, list):
        kept = [
            item
            for item in include
            if not (isinstance(item, str) and item.startswith("reasoning."))
        ]
        if kept:
            filtered["include"] = kept
        else:
            filtered.pop("include", None)
    return filtered


def filter_request_payload(
    payload: dict,
    mode: str,
    *,
    protocol: str | None = None,
) -> dict:
    """Return a normalized copy of an Anthropic or Responses request body."""
    validate_filter_mode(mode)
    filtered = _strip_reasoning_controls(payload)
    if mode == OMIT_REASONING:
        return filtered

    effort = mode.removeprefix(REASONING_EFFORT_PREFIX)
    if protocol is None:
        protocol = "openai-responses" if "input" in payload else "anthropic-messages"
    if protocol == "openai-responses":
        filtered["reasoning"] = {"effort": effort}
        return filtered
    if protocol != "anthropic-messages":
        raise ValueError(f"unsupported request protocol {protocol!r}")

    if effort == "none":
        filtered["thinking"] = {"type": "disabled"}
        return filtered
    filtered["thinking"] = {"type": "adaptive"}
    output_config = filtered.setdefault("output_config", {})
    if not isinstance(output_config, dict):
        raise ValueError("output_config must be an object")
    output_config["effort"] = effort
    return filtered


def upstream_request_path(upstream: SplitResult, incoming_path: str) -> str:
    """Join an agent request path to an upstream base URL path."""
    path, separator, query = incoming_path.partition("?")
    prefix = upstream.path.rstrip("/")
    joined = f"{prefix}/{path.lstrip('/')}" or "/"
    return f"{joined}{separator}{query}" if separator else joined


class RequestFilterHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, server_address, handler, *, upstream: str, mode: str):
        filter_mode = validate_filter_mode(mode)
        super().__init__(server_address, handler)
        self.upstream_url = upstream
        self.upstream = urlsplit(upstream)
        self.filter_mode = filter_mode
        if self.upstream.scheme not in {"http", "https"} or not self.upstream.hostname:
            raise ValueError(f"invalid upstream URL: {upstream!r}")


class RequestFilterHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: RequestFilterHTTPServer

    def _filtered_body(self) -> bytes | None:
        length = int(self.headers.get("content-length", "0"))
        if length == 0:
            return None
        body = self.rfile.read(length)
        content_type = self.headers.get("content-type", "").lower()
        if "json" not in content_type:
            return body
        payload = json.loads(body)
        if not isinstance(payload, dict):
            raise ValueError("request JSON body must be an object")
        protocol = request_protocol(self.path)
        if protocol is None:
            return body
        filtered = filter_request_payload(
            payload,
            self.server.filter_mode,
            protocol=protocol,
        )
        return json.dumps(filtered, separators=(",", ":")).encode()

    def _forward(self) -> None:
        connection: http.client.HTTPConnection | None = None
        try:
            body = self._filtered_body()
            upstream = self.server.upstream
            host = upstream.hostname
            if host is None:
                raise ValueError("upstream URL has no hostname")
            port = upstream.port or (443 if upstream.scheme == "https" else 80)
            if upstream.scheme == "https":
                connection = http.client.HTTPSConnection(
                    host,
                    port,
                    context=ssl.create_default_context(),
                    timeout=300,
                )
            else:
                connection = http.client.HTTPConnection(
                    host,
                    port,
                    timeout=300,
                )
            headers = {
                key: value
                for key, value in self.headers.items()
                if key.lower() not in _HOP_BY_HOP_HEADERS
            }
            connection.request(
                self.command,
                upstream_request_path(upstream, self.path),
                body=body,
                headers=headers,
            )
            response = connection.getresponse()
            self.send_response(response.status, response.reason)
            for key, value in response.getheaders():
                if key.lower() not in _HOP_BY_HOP_HEADERS:
                    self.send_header(key, value)
            self.send_header("Connection", "close")
            self.end_headers()
            while chunk := response.read(64 * 1024):
                self.wfile.write(chunk)
                self.wfile.flush()
        except Exception as exc:
            message = f"BenchFlow request filter failed: {type(exc).__name__}: {exc}\n"
            encoded = message.encode()
            self.send_response(502)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(encoded)
        finally:
            self.close_connection = True
            if connection is not None:
                connection.close()

    def do_GET(self) -> None:
        if self.path == HEALTH_PATH:
            encoded = json.dumps(
                {
                    "mode": self.server.filter_mode,
                    "upstream": self.server.upstream_url,
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)
            return
        self._forward()

    do_DELETE = _forward
    do_PATCH = _forward
    do_POST = _forward
    do_PUT = _forward

    def log_message(self, format: str, *args: Any) -> None:
        return


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--mode", type=validate_filter_mode, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=17891)
    args = parser.parse_args()

    server = RequestFilterHTTPServer(
        (args.host, args.port),
        RequestFilterHandler,
        upstream=args.upstream,
        mode=args.mode,
    )
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
