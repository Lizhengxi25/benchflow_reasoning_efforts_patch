"""Provider-bound HTTP filtering and wire-body capture.

This proxy sits between BenchFlow's host LiteLLM gateway and the real provider.
It records the complete body LiteLLM intended to send, applies the requested
reasoning normalization at the final outbound boundary, records the exact body
that is forwarded, and streams the provider response back unchanged while
persisting its raw body bytes.
"""

from __future__ import annotations

import base64
import copy
import http.client
import json
import os
import re
import ssl
import threading
import time
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import SplitResult, parse_qsl, urlencode, urlsplit, urlunsplit

PASSTHROUGH = "passthrough"
OMIT_REASONING = "omit-reasoning"
REASONING_EFFORT_PREFIX = "reasoning-effort:"
PROVIDER_REQUEST_FILTER_ENV = "BENCHFLOW_PROVIDER_REQUEST_FILTER"
SUPPORTED_REASONING_EFFORTS = frozenset(
    {"none", "minimal", "low", "medium", "high", "xhigh", "max"}
)
HEALTH_PATH = "/_benchflow/provider-wire-health"
DEFAULT_UPSTREAM_TIMEOUT_SEC = 115_200.0

_HOP_BY_HOP_HEADERS = frozenset(
    {
        "connection",
        "content-length",
        "host",
        "proxy-connection",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)
_SENSITIVE_NAME_PARTS = (
    "api-key",
    "apikey",
    "auth",
    "authorization",
    "cookie",
    "credential",
    "key",
    "password",
    "proxy-authorization",
    "secret",
    "signature",
    "token",
)
_REDACTED = "<redacted>"


def validate_filter_mode(mode: str) -> str:
    """Validate and return a provider request filter mode."""
    if mode in {PASSTHROUGH, OMIT_REASONING}:
        return mode
    if mode.startswith(REASONING_EFFORT_PREFIX):
        effort = mode.removeprefix(REASONING_EFFORT_PREFIX)
        if effort in SUPPORTED_REASONING_EFFORTS:
            return mode
    raise ValueError(f"unsupported provider request filter {mode!r}")


def request_protocol(path: str) -> str | None:
    """Identify model-generation endpoints without matching helper endpoints."""
    normalized = path.partition("?")[0].rstrip("/")
    if normalized.endswith("/responses"):
        return "openai-responses"
    if normalized.endswith("/chat/completions"):
        return "openai-completions"
    if normalized.endswith("/messages"):
        return "anthropic-messages"
    return None


def _without_reasoning_controls(payload: dict[str, Any]) -> dict[str, Any]:
    filtered = copy.deepcopy(payload)
    for key in ("include_reasoning", "reasoning", "reasoning_effort", "thinking"):
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

    for body_key in ("extra_body", "litellm_extra_body"):
        body = filtered.get(body_key)
        if not isinstance(body, dict):
            continue
        for key in ("include_reasoning", "reasoning", "reasoning_effort", "thinking"):
            body.pop(key, None)
        nested_output_config = body.get("output_config")
        if isinstance(nested_output_config, dict):
            nested_output_config.pop("effort", None)
            if not nested_output_config:
                body.pop("output_config", None)
        if not body:
            filtered.pop(body_key, None)
    return filtered


def filter_request_payload(
    payload: dict[str, Any],
    mode: str,
    *,
    protocol: str,
) -> dict[str, Any]:
    """Normalize one provider-bound request body at the outbound boundary."""
    validate_filter_mode(mode)
    if mode == PASSTHROUGH:
        return copy.deepcopy(payload)

    filtered = _without_reasoning_controls(payload)
    if mode == OMIT_REASONING:
        return filtered

    effort = mode.removeprefix(REASONING_EFFORT_PREFIX)
    if protocol in {"openai-responses", "openai-completions"}:
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
    """Join an incoming path to the configured provider base URL path."""
    path, separator, query = incoming_path.partition("?")
    prefix = upstream.path.rstrip("/")
    joined = f"{prefix}/{path.lstrip('/')}" or "/"
    return f"{joined}{separator}{query}" if separator else joined


def _is_sensitive_name(name: str) -> bool:
    normalized = name.lower().replace("_", "-")
    return any(part in normalized for part in _SENSITIVE_NAME_PARTS)


def _sanitize_headers(headers: Any) -> dict[str, str]:
    return {
        str(key): _REDACTED if _is_sensitive_name(str(key)) else str(value)
        for key, value in headers
    }


def _sanitize_url(upstream: SplitResult, incoming_path: str) -> str:
    path_with_query = upstream_request_path(upstream, incoming_path)
    path, separator, query = path_with_query.partition("?")
    sanitized_query = ""
    if separator:
        sanitized_query = urlencode(
            [
                (key, _REDACTED if _is_sensitive_name(key) else value)
                for key, value in parse_qsl(query, keep_blank_values=True)
            ]
        )
    host = upstream.hostname or ""
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    if upstream.port is not None:
        host = f"{host}:{upstream.port}"
    return urlunsplit((upstream.scheme, host, path, sanitized_query, ""))


def _sanitize_error(error: BaseException, secrets: list[str]) -> str:
    message = f"{type(error).__name__}: {error}"
    for secret in secrets:
        if secret:
            message = message.replace(secret, _REDACTED)
    return re.sub(
        r"(?i)\b(bearer|basic)\s+[^\s,;]+",
        rf"\1 {_REDACTED}",
        message,
    )


def _write_private_bytes(path: Path, data: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    os.fchmod(descriptor, 0o600)
    with os.fdopen(descriptor, "wb", buffering=0) as stream:
        stream.write(data)


def _write_private_json(path: Path, payload: Any) -> None:
    encoded = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode()
    temporary = path.with_name(f".{path.name}.tmp")
    _write_private_bytes(temporary, encoded)
    os.replace(temporary, path)
    os.chmod(path, 0o600)


def _parse_json_body(body: bytes, content_type: str) -> Any | None:
    if not body or "json" not in content_type.lower():
        return None
    try:
        return json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None


class WireCapture:
    """Incrementally persist one provider exchange without buffering the response."""

    def __init__(
        self,
        request_dir: Path,
        *,
        sequence: int,
        method: str,
        url: str,
        logical_headers: Any,
        filter_mode: str,
        protocol: str | None,
        sensitive_values: list[str],
    ) -> None:
        self.request_dir = request_dir
        self._started_monotonic = time.monotonic()
        self._sensitive_values = sensitive_values
        self._response_stream: Any | None = None
        self._response_bytes = 0
        started_at = datetime.now(UTC).isoformat()
        self._metadata: dict[str, Any] = {
            "capture_version": 1,
            "sequence": sequence,
            "filter_mode": filter_mode,
            "protocol": protocol,
            "started_at": started_at,
            "finished_at": None,
            "duration_ms": None,
            "transport_complete": False,
            "error": None,
            "timing": {
                "started_at": started_at,
                "response_headers_at": None,
                "first_response_byte_at": None,
                "finished_at": None,
                "time_to_headers_ms": None,
                "time_to_first_byte_ms": None,
                "duration_ms": None,
            },
            "request": {
                "method": method,
                "url": url,
                "logical_headers": _sanitize_headers(logical_headers),
                "provider_headers": None,
                "logical_body_bytes": None,
                "provider_body_bytes": None,
            },
            "response": None,
        }
        self._write_metadata()

    def _write_metadata(self) -> None:
        _write_private_json(self.request_dir / "metadata.json", self._metadata)

    def record_logical_request(self, body: bytes, content_type: str) -> None:
        _write_private_bytes(self.request_dir / "logical_request.body", body)
        payload = _parse_json_body(body, content_type)
        if payload is not None:
            _write_private_json(self.request_dir / "logical_request.json", payload)
        self._metadata["request"]["logical_body_bytes"] = len(body)
        self._write_metadata()

    def record_provider_request(
        self,
        body: bytes,
        content_type: str,
        provider_headers: Any,
    ) -> None:
        _write_private_bytes(self.request_dir / "provider_request.body", body)
        payload = _parse_json_body(body, content_type)
        if payload is not None:
            _write_private_json(self.request_dir / "provider_request.json", payload)
        self._metadata["request"]["provider_headers"] = _sanitize_headers(
            provider_headers
        )
        self._metadata["request"]["provider_body_bytes"] = len(body)
        self._write_metadata()

    def start_response(
        self,
        *,
        status: int,
        reason: str,
        headers: Any,
        content_type: str,
    ) -> None:
        response_headers_at = datetime.now(UTC).isoformat()
        elapsed_ms = round(
            (time.monotonic() - self._started_monotonic) * 1000,
            3,
        )
        suffix = "sse" if "text/event-stream" in content_type.lower() else "body"
        response_path = self.request_dir / f"provider_response.{suffix}"
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(response_path, flags, 0o600)
        os.fchmod(descriptor, 0o600)
        self._response_stream = os.fdopen(descriptor, "wb", buffering=0)
        self._metadata["timing"]["response_headers_at"] = response_headers_at
        self._metadata["timing"]["time_to_headers_ms"] = elapsed_ms
        self._metadata["response"] = {
            "status": status,
            "reason": reason,
            "headers": _sanitize_headers(headers),
            "content_type": content_type,
            "body_bytes": 0,
            "file": response_path.name,
        }
        self._write_metadata()

    def write_response(self, chunk: bytes) -> None:
        if self._response_stream is None:
            return
        if self._response_bytes == 0:
            first_byte_at = datetime.now(UTC).isoformat()
            self._metadata["timing"]["first_response_byte_at"] = first_byte_at
            self._metadata["timing"]["time_to_first_byte_ms"] = round(
                (time.monotonic() - self._started_monotonic) * 1000,
                3,
            )
        self._response_stream.write(chunk)
        self._response_bytes += len(chunk)

    def finish(self) -> None:
        self._close_response_stream()
        self._metadata["transport_complete"] = True
        self._finish_metadata()

    def fail(self, error: BaseException) -> None:
        self._close_response_stream()
        self._metadata["error"] = _sanitize_error(error, self._sensitive_values)
        self._finish_metadata()

    def _close_response_stream(self) -> None:
        if self._response_stream is not None:
            self._response_stream.close()
            self._response_stream = None
        if self._metadata["response"] is not None:
            self._metadata["response"]["body_bytes"] = self._response_bytes

    def _finish_metadata(self) -> None:
        finished_at = datetime.now(UTC).isoformat()
        duration_ms = round(
            (time.monotonic() - self._started_monotonic) * 1000,
            3,
        )
        self._metadata["finished_at"] = finished_at
        self._metadata["duration_ms"] = duration_ms
        self._metadata["timing"]["finished_at"] = finished_at
        self._metadata["timing"]["duration_ms"] = duration_ms
        self._write_metadata()


class _NullCapture:
    """No-op sink used when filtering is enabled without raw-body capture."""

    def record_logical_request(self, body: bytes, content_type: str) -> None:
        return

    def record_provider_request(
        self,
        body: bytes,
        content_type: str,
        provider_headers: Any,
    ) -> None:
        return

    def start_response(
        self,
        *,
        status: int,
        reason: str,
        headers: Any,
        content_type: str,
    ) -> None:
        return

    def write_response(self, chunk: bytes) -> None:
        return

    def finish(self) -> None:
        return

    def fail(self, error: BaseException) -> None:
        return


class ProviderWireHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        *,
        upstream: str,
        mode: str,
        capture_dir: Path | None,
        upstream_timeout_sec: float,
    ) -> None:
        self.upstream_url = upstream
        self.upstream = urlsplit(upstream)
        self.filter_mode = validate_filter_mode(mode)
        self.capture_dir = Path(capture_dir) if capture_dir is not None else None
        self.upstream_timeout_sec = upstream_timeout_sec
        self._capture_lock = threading.Lock()
        self._capture_sequence = 0
        if self.upstream.scheme not in {"http", "https"} or not self.upstream.hostname:
            raise ValueError(f"invalid upstream URL: {upstream!r}")
        if self.capture_dir is not None:
            self.capture_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            os.chmod(self.capture_dir, 0o700)
            existing_sequences = [
                int(path.name)
                for path in self.capture_dir.iterdir()
                if path.is_dir() and re.fullmatch(r"\d{6}", path.name)
            ]
            self._capture_sequence = max(existing_sequences, default=0)
        super().__init__(server_address, ProviderWireHandler)

    def new_capture(self, handler: ProviderWireHandler) -> WireCapture | _NullCapture:
        if self.capture_dir is None:
            return _NullCapture()
        with self._capture_lock:
            while True:
                self._capture_sequence += 1
                sequence = self._capture_sequence
                request_dir = self.capture_dir / f"{sequence:06d}"
                try:
                    request_dir.mkdir(mode=0o700)
                    os.chmod(request_dir, 0o700)
                    break
                except FileExistsError:
                    continue
        header_items = list(handler.headers.items())
        sensitive_values = [
            str(value) for key, value in header_items if _is_sensitive_name(str(key))
        ]
        if self.upstream.username:
            sensitive_values.append(self.upstream.username)
        if self.upstream.password:
            sensitive_values.append(self.upstream.password)
        return WireCapture(
            request_dir,
            sequence=sequence,
            method=handler.command,
            url=_sanitize_url(self.upstream, handler.path),
            logical_headers=header_items,
            filter_mode=self.filter_mode,
            protocol=request_protocol(handler.path),
            sensitive_values=sensitive_values,
        )


class ProviderWireHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: ProviderWireHTTPServer

    def _read_body(self) -> bytes:
        length = int(self.headers.get("content-length", "0"))
        return self.rfile.read(length) if length else b""

    def _filtered_body(self, body: bytes) -> bytes:
        protocol = request_protocol(self.path)
        if (
            not body
            or protocol is None
            or self.server.filter_mode == PASSTHROUGH
            or "json" not in self.headers.get("content-type", "").lower()
        ):
            return body
        payload = json.loads(body)
        if not isinstance(payload, dict):
            raise ValueError("provider request JSON body must be an object")
        filtered = filter_request_payload(
            payload,
            self.server.filter_mode,
            protocol=protocol,
        )
        return json.dumps(filtered, ensure_ascii=False, separators=(",", ":")).encode()

    def _provider_headers(self) -> dict[str, str]:
        headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower() not in _HOP_BY_HOP_HEADERS
        }
        if self.server.upstream.username is not None:
            password = self.server.upstream.password or ""
            credentials = base64.b64encode(
                f"{self.server.upstream.username}:{password}".encode()
            ).decode()
            headers["Authorization"] = f"Basic {credentials}"
        return headers

    def _send_response_headers(
        self,
        *,
        status: int,
        reason: str,
        headers: list[tuple[str, str]],
    ) -> None:
        self.send_response(status, reason)
        for key, value in headers:
            if key.lower() not in _HOP_BY_HOP_HEADERS:
                self.send_header(key, value)
        self.send_header("Connection", "close")
        self.end_headers()

    def _forward(self) -> None:
        connection: http.client.HTTPConnection | None = None
        capture = self.server.new_capture(self)
        response_headers_sent = False
        try:
            logical_body = self._read_body()
            content_type = self.headers.get("content-type", "")
            capture.record_logical_request(logical_body, content_type)
            provider_body = self._filtered_body(logical_body)
            provider_headers = self._provider_headers()
            capture.record_provider_request(
                provider_body,
                content_type,
                provider_headers.items(),
            )

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
                    timeout=self.server.upstream_timeout_sec,
                )
            else:
                connection = http.client.HTTPConnection(
                    host,
                    port,
                    timeout=self.server.upstream_timeout_sec,
                )
            connection.request(
                self.command,
                upstream_request_path(upstream, self.path),
                body=provider_body,
                headers=provider_headers,
            )
            response = connection.getresponse()
            response_headers = response.getheaders()
            response_content_type = response.getheader("content-type", "")
            capture.start_response(
                status=response.status,
                reason=response.reason,
                headers=response_headers,
                content_type=response_content_type,
            )
            self._send_response_headers(
                status=response.status,
                reason=response.reason,
                headers=response_headers,
            )
            response_headers_sent = True
            read_response = getattr(response, "read1", response.read)
            while True:
                try:
                    chunk = read_response(64 * 1024)
                except http.client.IncompleteRead as exc:
                    if exc.partial:
                        capture.write_response(exc.partial)
                        self.wfile.write(exc.partial)
                        self.wfile.flush()
                    raise
                if not chunk:
                    if response.length not in {None, 0}:
                        raise http.client.IncompleteRead(b"", response.length)
                    break
                capture.write_response(chunk)
                self.wfile.write(chunk)
                self.wfile.flush()
            capture.finish()
        except Exception as exc:
            capture.fail(exc)
            if not response_headers_sent:
                message = (
                    f"BenchFlow provider wire proxy failed: {type(exc).__name__}: "
                    f"{exc}\n"
                ).encode()
                self.send_response(502)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(message)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(message)
        finally:
            self.close_connection = True
            if connection is not None:
                connection.close()

    def do_GET(self) -> None:
        if self.path == HEALTH_PATH:
            encoded = json.dumps(
                {
                    "mode": self.server.filter_mode,
                    "upstream": _sanitize_url(self.server.upstream, ""),
                    "capture_dir": (
                        str(self.server.capture_dir)
                        if self.server.capture_dir is not None
                        else None
                    ),
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


class ProviderWireProxy:
    """Lifecycle wrapper for one loopback provider-wire server."""

    def __init__(
        self,
        *,
        upstream: str,
        mode: str,
        capture_dir: Path | None = None,
        upstream_timeout_sec: float = DEFAULT_UPSTREAM_TIMEOUT_SEC,
    ) -> None:
        self.server = ProviderWireHTTPServer(
            ("127.0.0.1", 0),
            upstream=upstream,
            mode=mode,
            capture_dir=capture_dir,
            upstream_timeout_sec=upstream_timeout_sec,
        )
        self.thread = threading.Thread(
            target=self.server.serve_forever,
            name="benchflow-provider-wire",
            daemon=True,
        )
        self.thread.start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}"

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
