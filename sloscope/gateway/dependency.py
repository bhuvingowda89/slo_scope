from __future__ import annotations

import argparse
import asyncio
import json
import threading
import time
from dataclasses import dataclass
from typing import Optional

from sloscope.gateway.http import read_http_request, split_base_url, write_json_response


@dataclass
class DependencyState:
    service_id: str
    port: int
    maximum_delay_ms: int
    delay_ms: int = 0


class SyntheticDependencyService:
    def __init__(self, host: str = "127.0.0.1", port: int = 8091, maximum_delay_ms: int = 500, service_id: str = "synthetic_dependency") -> None:
        self.host = host
        self.port = port
        self.state = DependencyState(service_id, port, maximum_delay_ms)
        self._lock = threading.Lock()
        self._server: Optional[asyncio.AbstractServer] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None

    def set_delay(self, delay_ms: int) -> None:
        if delay_ms < 0:
            raise ValueError("delay_ms must be non-negative")
        if delay_ms > self.state.maximum_delay_ms:
            raise ValueError("delay_ms exceeds maximum_dependency_delay_ms")
        with self._lock:
            self.state.delay_ms = int(delay_ms)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "service_id": self.state.service_id,
                "port": self.state.port,
                "delay_ms": self.state.delay_ms,
                "maximum_delay_ms": self.state.maximum_delay_ms,
                "status": "ok",
            }

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            request = await read_http_request(reader)
            if request is None:
                return
            if request.method == "GET" and request.path in {"/health", "/control/status"}:
                await write_json_response(writer, 200, self.snapshot())
                return
            if request.method == "POST" and request.path == "/control/delay":
                payload = json.loads(request.body.decode("utf-8") or "{}")
                self.set_delay(int(payload.get("delay_ms", 0)))
                await write_json_response(writer, 200, self.snapshot())
                return
            if request.method == "GET" and request.path == "/dependency":
                started = time.monotonic()
                delay_ms = int(self.snapshot()["delay_ms"])
                if delay_ms:
                    await asyncio.sleep(delay_ms / 1000.0)
                ended = time.monotonic()
                await write_json_response(writer, 200, {**self.snapshot(), "dependency_duration": ended - started})
                return
            await write_json_response(writer, 404, {"error": "not found"})
        except Exception as exc:
            await write_json_response(writer, 400, {"error": str(exc)})
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
        raise RuntimeError("dependency service failed to start")

    def stop(self) -> None:
        self.set_delay(0)
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
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--maximum-delay-ms", type=int, default=500)
    args = parser.parse_args(argv)
    service = SyntheticDependencyService(args.host, args.port, args.maximum_delay_ms)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(service._start_async())
    try:
        loop.run_forever()
    finally:
        service.set_delay(0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
