from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit

from sloscope.gateway.http import read_http_request, split_base_url, write_json_response, write_response
from sloscope.runtime.llamacpp import content_from_chunk, parse_sse_payload


def _json_request(method: str, url: str, timeout: float, payload: Optional[dict] = None) -> tuple[int, Any]:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
        return int(resp.status), json.loads(body) if body else {}


def _raw_get(url: str, timeout: float) -> tuple[int, bytes, str]:
    req = urllib.request.Request(url, method="GET", headers={"Accept": "*/*"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return int(resp.status), resp.read(), resp.headers.get("Content-Type", "text/plain")


def _span(trace_id: str, span_id: str, parent_span_id: Optional[str], span_name: str, start: float, end: float, status: str, attributes: dict) -> dict:
    return {
        "trace_id": trace_id,
        "span_id": span_id,
        "parent_span_id": parent_span_id,
        "span_name": span_name,
        "start_time": start,
        "end_time": end,
        "duration": end - start,
        "status": status,
        "attributes": json.dumps(attributes, sort_keys=True, separators=(",", ":")),
    }


def deterministic_id(*parts: object, length: int = 16) -> str:
    return hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()[:length]


async def _read_status_and_headers(reader: asyncio.StreamReader) -> tuple[int, dict[str, str]]:
    head = await reader.readuntil(b"\r\n\r\n")
    lines = head.decode("iso-8859-1").split("\r\n")
    if not lines or not lines[0].startswith("HTTP/"):
        raise RuntimeError("malformed upstream HTTP response")
    parts = lines[0].split(" ", 2)
    status = int(parts[1])
    headers: dict[str, str] = {}
    for line in lines[1:]:
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        headers[key.strip().lower()] = value.strip()
    return status, headers


async def _async_http_request(method: str, url: str, timeout: float, payload: Optional[dict | bytes] = None, headers: Optional[dict[str, str]] = None) -> tuple[int, dict[str, str], bytes]:
    parsed = urlsplit(url)
    host, port = split_base_url(f"{parsed.scheme}://{parsed.netloc}")
    path = parsed.path or "/"
    if parsed.query:
        path = f"{path}?{parsed.query}"
    body = b""
    if isinstance(payload, dict):
        body = json.dumps(payload).encode("utf-8")
    elif isinstance(payload, bytes):
        body = payload
    out_headers = {
        "Host": f"{host}:{port}",
        "Connection": "close",
        "Accept": "*/*",
        "Content-Length": str(len(body)),
    }
    if headers:
        out_headers.update(headers)
    reader, writer = await asyncio.wait_for(asyncio.open_connection(host, port), timeout=timeout)
    try:
        writer.write(f"{method} {path} HTTP/1.1\r\n".encode("ascii"))
        for key, value in out_headers.items():
            writer.write(f"{key}: {value}\r\n".encode("ascii"))
        writer.write(b"\r\n")
        writer.write(body)
        await writer.drain()
        status, resp_headers = await asyncio.wait_for(_read_status_and_headers(reader), timeout=timeout)
        if "content-length" in resp_headers:
            resp_body = await asyncio.wait_for(reader.readexactly(int(resp_headers["content-length"])), timeout=timeout)
        else:
            resp_body = await asyncio.wait_for(reader.read(), timeout=timeout)
        return status, resp_headers, resp_body
    finally:
        writer.close()
        await writer.wait_closed()


async def _async_json_request(method: str, url: str, timeout: float, payload: Optional[dict] = None) -> tuple[int, Any]:
    headers = {"Accept": "application/json"}
    if payload is not None:
        headers["Content-Type"] = "application/json"
    status, _, body = await _async_http_request(method, url, timeout, payload, headers)
    return status, json.loads(body.decode("utf-8")) if body else {}


async def _open_streaming_post(url: str, timeout: float, body: bytes, headers: dict[str, str]) -> tuple[int, dict[str, str], asyncio.StreamReader, asyncio.StreamWriter]:
    parsed = urlsplit(url)
    host, port = split_base_url(f"{parsed.scheme}://{parsed.netloc}")
    path = parsed.path or "/"
    reader, writer = await asyncio.wait_for(asyncio.open_connection(host, port), timeout=timeout)
    out_headers = {
        "Host": f"{host}:{port}",
        "Connection": "close",
        "Content-Length": str(len(body)),
    }
    out_headers.update(headers)
    writer.write(f"POST {path} HTTP/1.1\r\n".encode("ascii"))
    for key, value in out_headers.items():
        writer.write(f"{key}: {value}\r\n".encode("ascii"))
    writer.write(b"\r\n")
    writer.write(body)
    await writer.drain()
    status, resp_headers = await asyncio.wait_for(_read_status_and_headers(reader), timeout=timeout)
    return status, resp_headers, reader, writer


async def _stream_sse_body(
    reader: asyncio.StreamReader,
    headers: dict[str, str],
    downstream: asyncio.StreamWriter,
    timeout: float,
    on_line,
) -> None:
    if headers.get("transfer-encoding", "").lower() == "chunked":
        while True:
            size_line = await asyncio.wait_for(reader.readline(), timeout=timeout)
            if not size_line:
                break
            size = int(size_line.split(b";", 1)[0].strip() or b"0", 16)
            if size == 0:
                await asyncio.wait_for(reader.readline(), timeout=timeout)
                break
            data = await asyncio.wait_for(reader.readexactly(size), timeout=timeout)
            await asyncio.wait_for(reader.readexactly(2), timeout=timeout)
            for line in data.splitlines(keepends=True):
                on_line(line)
            downstream.write(data)
            await downstream.drain()
        return
    while True:
        raw = await asyncio.wait_for(reader.readline(), timeout=timeout)
        if not raw:
            break
        on_line(raw)
        downstream.write(raw)
        await downstream.drain()


class SLOScopeGateway:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8090,
        llama_base_url: str = "http://127.0.0.1:8080",
        dependency_base_url: str = "http://127.0.0.1:8091",
        timeout: float = 60.0,
    ) -> None:
        split_base_url(llama_base_url)
        split_base_url(dependency_base_url)
        self.host = host
        self.port = port
        self.llama_base_url = llama_base_url.rstrip("/")
        self.dependency_base_url = dependency_base_url.rstrip("/")
        self.timeout = timeout
        self._server: Optional[asyncio.AbstractServer] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self.request_timings: Dict[str, dict] = {}
        self.trace_rows: List[dict] = []
        self.reset_count = 0

    async def _proxy_json(self, writer: asyncio.StreamWriter, path: str) -> None:
        try:
            status, payload = await asyncio.to_thread(_json_request, "GET", f"{self.llama_base_url}{path}", self.timeout, None)
            await write_json_response(writer, status, payload)
        except Exception as exc:
            await write_json_response(writer, 502, {"error": str(exc)})

    async def _dependency_call(self) -> tuple[dict, float, float]:
        start = time.monotonic()
        status, payload = await _async_json_request("GET", f"{self.dependency_base_url}/dependency", self.timeout, None)
        end = time.monotonic()
        if status >= 400 or not isinstance(payload, dict):
            raise RuntimeError(f"dependency request failed: status={status}")
        return payload, start, end

    async def _ready_payload(self) -> tuple[int, dict]:
        gateway_alive = True
        dependency_reachable = False
        llama_reachable = False
        dependency_status: dict[str, Any] | None = None
        llama_status: dict[str, Any] | None = None
        try:
            status, payload = await _async_json_request("GET", f"{self.dependency_base_url}/control/status", min(self.timeout, 2.0))
            dependency_reachable = status == 200 and isinstance(payload, dict) and payload.get("status") == "ok"
            dependency_status = payload if isinstance(payload, dict) else {"payload": payload}
        except Exception as exc:
            dependency_status = {"error": str(exc)}
        try:
            status, payload = await _async_json_request("GET", f"{self.llama_base_url}/health", min(self.timeout, 2.0))
            llama_reachable = status == 200 and isinstance(payload, dict) and str(payload.get("status", "")).lower() == "ok"
            llama_status = payload if isinstance(payload, dict) else {"payload": payload}
        except Exception as exc:
            llama_status = {"error": str(exc)}
        ready = gateway_alive and dependency_reachable and llama_reachable
        return 200 if ready else 503, {
            "status": "ok" if ready else "unavailable",
            "gateway_alive": gateway_alive,
            "dependency_reachable": dependency_reachable,
            "llama_reachable": llama_reachable,
            "dependency": dependency_status,
            "llama": llama_status,
        }

    def reset(self) -> dict:
        with self._lock:
            cleared_request_timings = len(self.request_timings)
            cleared_trace_rows = len(self.trace_rows)
            self.request_timings.clear()
            self.trace_rows.clear()
            self.reset_count += 1
            reset_count = self.reset_count
        return {
            "status": "ok",
            "gateway": "sloscope",
            "reset_count": reset_count,
            "cleared_request_timings": cleared_request_timings,
            "cleared_trace_rows": cleared_trace_rows,
        }

    async def _handle_completion(self, request_id: str, request_body: bytes, writer: asyncio.StreamWriter) -> None:
        receive = time.monotonic()
        trace_id = deterministic_id("trace", request_id, receive, length=32)
        root_span = deterministic_id(trace_id, "gateway")
        dep_span = deterministic_id(trace_id, "dependency")
        llama_span = deterministic_id(trace_id, "llama")
        status = "ok"
        dependency_payload: dict = {}
        llama_dispatch = None
        llama_first_token = None
        completion = None
        dep_status = "error"
        llama_status = "error"
        response_started = False
        try:
            dependency_payload, dep_start, dep_end = await self._dependency_call()
            dep_status = "ok"
            llama_dispatch = time.monotonic()
            upstream_status, upstream_headers, upstream_reader, upstream_writer = await _open_streaming_post(
                f"{self.llama_base_url}/v1/completions",
                self.timeout,
                request_body,
                {"Content-Type": "application/json", "Accept": "text/event-stream", "X-SLOScope-Request-Id": request_id},
            )
            if upstream_status >= 400:
                upstream_writer.close()
                await upstream_writer.wait_closed()
                raise RuntimeError(f"llama upstream returned status={upstream_status}")
            llama_status = "ok"
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n")
            response_started = True
            await writer.drain()
            try:
                def observe(raw: bytes) -> None:
                    nonlocal llama_first_token
                    parsed = parse_sse_payload(raw)
                    if isinstance(parsed, dict) and llama_first_token is None and content_from_chunk(parsed):
                        llama_first_token = time.monotonic()

                await _stream_sse_body(upstream_reader, upstream_headers, writer, self.timeout, observe)
            finally:
                upstream_writer.close()
                await upstream_writer.wait_closed()
            completion = time.monotonic()
        except Exception as exc:
            status = "error"
            if dep_status == "ok" and llama_dispatch is not None:
                llama_status = "error"
            completion = time.monotonic()
            if llama_dispatch is None:
                llama_dispatch = completion
            if "dep_start" not in locals():
                dep_start = receive
                dep_end = completion
            payload = {"error": str(exc)}
            if not response_started:
                await write_response(writer, 502, json.dumps(payload).encode("utf-8"), {"Content-Type": "application/json"})
        finally:
            timing = {
                "request_id": request_id,
                "gateway_receive_time": receive,
                "dependency_start_time": dep_start,
                "dependency_end_time": dep_end,
                "llama_dispatch_time": llama_dispatch,
                "llama_first_token_time": llama_first_token,
                "gateway_completion_time": completion,
                "dependency_duration": dep_end - dep_start,
                "dependency_payload": dependency_payload,
                "status": status,
            }
            spans = [
                _span(trace_id, root_span, None, "gateway request", receive, completion, status, {"request_id": request_id, "gateway_port": self.port}),
                _span(trace_id, dep_span, root_span, "dependency call", dep_start, dep_end, dep_status, {"request_id": request_id, "dependency_port": split_base_url(self.dependency_base_url)[1]}),
                _span(trace_id, llama_span, root_span, "llama call", llama_dispatch, completion, llama_status, {"request_id": request_id, "llama_base_url": self.llama_base_url}),
            ]
            with self._lock:
                self.request_timings[request_id] = timing
                self.trace_rows.extend(spans)

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            request = await read_http_request(reader)
            if request is None:
                return
            if request.method == "GET" and request.path == "/health":
                await write_json_response(writer, 200, {"status": "ok", "gateway_port": self.port})
                return
            if request.method == "GET" and request.path == "/ready":
                status, payload = await self._ready_payload()
                await write_json_response(writer, status, payload)
                return
            if request.method == "POST" and request.path == "/sloscope/reset":
                await write_json_response(writer, 200, self.reset())
                return
            if request.method == "GET" and request.path == "/v1/models":
                await self._proxy_json(writer, request.path)
                return
            if request.method == "GET" and request.path == "/metrics":
                try:
                    status, body, content_type = await asyncio.to_thread(_raw_get, f"{self.llama_base_url}/metrics", self.timeout)
                    await write_response(writer, status, body, {"Content-Type": content_type})
                except Exception as exc:
                    await write_json_response(writer, 502, {"error": str(exc)})
                return
            if request.method == "GET" and request.path.startswith("/sloscope/requests/"):
                rid = request.path.rsplit("/", 1)[-1]
                with self._lock:
                    payload = self.request_timings.get(rid)
                await write_json_response(writer, 200 if payload else 404, payload or {"error": "not found"})
                return
            if request.method == "GET" and request.path == "/sloscope/traces":
                with self._lock:
                    payload = {"spans": list(self.trace_rows)}
                await write_json_response(writer, 200, payload)
                return
            if request.method == "POST" and request.path == "/v1/completions":
                request_id = request.headers.get("x-sloscope-request-id") or deterministic_id("request", time.monotonic())
                await self._handle_completion(request_id, request.body, writer)
                return
            await write_json_response(writer, 404, {"error": "not found"})
        finally:
            writer.close()
            await writer.wait_closed()

    async def _start_async(self) -> None:
        self._server = await asyncio.start_server(self._handle, self.host, self.port)

    def start(self) -> None:
        self._loop = asyncio.new_event_loop()

        def run() -> None:
            assert self._loop is not None
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(self._start_async())
            self._loop.run_forever()

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()
        deadline = time.time() + 2.0
        while time.time() < deadline:
            if self._server is not None:
                return
            time.sleep(0.01)
        raise RuntimeError("gateway failed to start")

    def stop(self) -> None:
        if self._loop is None:
            return
        server = self._server

        async def close() -> None:
            if server is not None:
                server.close()
                await server.wait_closed()

        fut = asyncio.run_coroutine_threadsafe(close(), self._loop)
        fut.result(timeout=2.0)
        self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=2.0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--llama-base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--dependency-base-url", default="http://127.0.0.1:8091")
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args(argv)
    gateway = SLOScopeGateway(args.host, args.port, args.llama_base_url, args.dependency_base_url, args.timeout)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(gateway._start_async())
    loop.run_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
