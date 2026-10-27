"""Fan-out of events to every connected browser, plus the state a late joiner needs."""

from __future__ import annotations

import asyncio
import json
import struct

from fastapi import WebSocket

MAP_CAP = 2_500_000  # points kept for late joiners


def _plain(o):
    """JSON fallback for numpy scalars and arrays."""
    return o.tolist() if hasattr(o, "tolist") else str(o)


class Hub:
    def __init__(self) -> None:
        self.clients: set[WebSocket] = set()
        self.map_chunks: list[bytes] = []  # ready-to-send binary messages
        self.map_total = 0
        self.last: dict[str, dict] = {}  # latest message of each replayable type

    async def _send(self, ws: WebSocket, *, text: str | None = None, data: bytes | None = None):
        try:
            if text is not None:
                await ws.send_text(text)
            else:
                await ws.send_bytes(data)  # type: ignore[arg-type]
        except Exception:
            self.clients.discard(ws)

    async def json(self, msg: dict, replay: bool = False) -> None:
        if replay:
            self.last[msg["type"]] = msg
        text = json.dumps(msg, separators=(",", ":"), default=_plain)
        await asyncio.gather(*(self._send(c, text=text) for c in list(self.clients)))

    async def map_chunk(self, n: int, points: bytes, colors: bytes) -> None:
        data = b"M" + struct.pack("<I", n) + points + colors
        if self.map_total + n <= MAP_CAP:
            self.map_chunks.append(data)
            self.map_total += n
        await asyncio.gather(*(self._send(c, data=data) for c in list(self.clients)))

    async def join(self, ws: WebSocket) -> None:
        self.clients.add(ws)
        for chunk in self.map_chunks:
            await self._send(ws, data=chunk)
        for msg in self.last.values():
            await self._send(ws, text=json.dumps(msg, separators=(",", ":")))
