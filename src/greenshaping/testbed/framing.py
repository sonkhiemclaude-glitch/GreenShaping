"""Frame format for the shaped transport (Sec. 4.2, 4.3).

Every frame is encrypted by TLS as a whole, so the observer sees only its length.
Layout: 1 byte type | 1 byte flags | 4 bytes payload length | payload | zero padding.
Flag bit 0 = "reply complete" (how a called site signals completion inside its
encrypted frames, Sec. 4.2). Frames are padded to a target wire size.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

HEADER = struct.Struct("!BBI")
CALL, REPLY, POLL, FILLER, START = 1, 2, 3, 4, 5
FLAG_COMPLETE = 0x01


@dataclass(frozen=True)
class Frame:
    ftype: int
    payload: bytes = b""
    complete: bool = False


def encode(frame: Frame, wire_size: int) -> bytes:
    """Encode and pad to exactly `wire_size` bytes (ValueError if the payload does not fit)."""
    if frame.ftype not in (CALL, REPLY, POLL, FILLER, START):
        raise ValueError(f"unknown frame type {frame.ftype}")
    need = HEADER.size + len(frame.payload)
    if need > wire_size:
        raise ValueError(f"payload of {len(frame.payload)} B does not fit a {wire_size} B frame")
    flags = FLAG_COMPLETE if frame.complete else 0
    return HEADER.pack(frame.ftype, flags, len(frame.payload)) + frame.payload + b"\x00" * (wire_size - need)


def decode(data: bytes) -> Frame:
    if len(data) < HEADER.size:
        raise ValueError("frame shorter than header")
    ftype, flags, n = HEADER.unpack_from(data)
    if HEADER.size + n > len(data):
        raise ValueError("declared payload length exceeds frame")
    return Frame(ftype, data[HEADER.size:HEADER.size + n], bool(flags & FLAG_COMPLETE))


def split_reply(payload: bytes, frame_size: int) -> list[Frame]:
    """Split a reply over frames of `frame_size` wire bytes; the last carries the completion flag."""
    cap = frame_size - HEADER.size
    if cap <= 0:
        raise ValueError("frame_size too small for the header")
    chunks = [payload[i:i + cap] for i in range(0, len(payload), cap)] or [b""]
    return [Frame(REPLY, c, complete=(k == len(chunks) - 1)) for k, c in enumerate(chunks)]
