from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Dict, Tuple
from urllib.parse import urlsplit


@dataclass
class HTTPRequest:
    method: str
    path: str
    headers: Dict[str, str]
    body: bytes


async def read_http_request(reader: asyncio.StreamReader) -> HTTPRequest | None:
    head = await reader.readuntil(b"\r\n\r\n")
    lines = head.decode("iso-8859-1").split("\r\n")
    if not lines or not lines[0]:
        return None
    method, raw_path, _ = lines[0].split(" ", 2)
    headers: Dict[str, str] = {}
    for line in lines[1:]:
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        headers[key.strip().lower()] = value.strip()
    length = int(headers.get("content-length", "0") or "0")
    body = await reader.readexactly(length) if length else b""
    return HTTPRequest(method.upper(), urlsplit(raw_path).path, headers, body)


async def write_response(writer: asyncio.StreamWriter, status: int, body: bytes = b"", headers: Dict[str, str] | None = None) -> None:
    reason = {200: "OK", 202: "Accepted", 400: "Bad Request", 404: "Not Found", 502: "Bad Gateway", 503: "Unavailable"}.get(status, "OK")
    out_headers = {"Content-Length": str(len(body)), "Connection": "close"}
    if headers:
        out_headers.update(headers)
    writer.write(f"HTTP/1.1 {status} {reason}\r\n".encode("ascii"))
    for key, value in out_headers.items():
        writer.write(f"{key}: {value}\r\n".encode("ascii"))
    writer.write(b"\r\n")
    writer.write(body)
    await writer.drain()


async def write_json_response(writer: asyncio.StreamWriter, status: int, payload: dict) -> None:
    await write_response(writer, status, json.dumps(payload, sort_keys=True).encode("utf-8"), {"Content-Type": "application/json"})


def parse_json_body(request: HTTPRequest) -> dict:
    if not request.body:
        return {}
    return json.loads(request.body.decode("utf-8"))


def split_base_url(url: str) -> Tuple[str, int]:
    parsed = urlsplit(url)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"}:
        raise ValueError("only localhost http URLs are supported")
    return parsed.hostname, parsed.port or 80
