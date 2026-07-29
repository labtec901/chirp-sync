"""Take-ID payload encoding.

The on-air payload is deliberately fixed and contains exactly one value::

    [ 40-bit take ID ][ CRC-16 ]

The checksum protects the ID but is not application data. Everything is
big-endian and byte aligned. This format is mirrored in
``webapp/chirpsync.js``.
"""

from __future__ import annotations

import secrets
import struct
from dataclasses import dataclass

TAKE_ID_BITS = 40
TAKE_ID_BYTES = TAKE_ID_BITS // 8
PAYLOAD_BYTES = TAKE_ID_BYTES + 2

# Crockford base32: no I, L, O or U, so nothing gets misread off a screen or
# mistyped into a filename.
_B32 = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def take_id_to_str(take_id: int) -> str:
    if not 0 <= take_id < (1 << TAKE_ID_BITS):
        raise ValueError("take id out of range")
    chars = []
    for i in range(TAKE_ID_BITS // 5):
        chars.append(_B32[(take_id >> (TAKE_ID_BITS - 5 * (i + 1))) & 0x1F])
    return "".join(chars)


def take_id_from_str(text: str) -> int:
    text = text.strip().upper().replace("-", "")
    # Accept the characters Crockford folds together.
    text = text.replace("I", "1").replace("L", "1").replace("O", "0").replace("U", "V")
    if len(text) != TAKE_ID_BITS // 5:
        raise ValueError("take id string must be 8 characters")
    value = 0
    for ch in text:
        idx = _B32.find(ch)
        if idx < 0:
            raise ValueError(f"invalid take id character {ch!r}")
        value = (value << 5) | idx
    return value


def new_take_id() -> int:
    return secrets.randbits(TAKE_ID_BITS)


def crc16(data: bytes) -> int:
    """CRC-16/CCITT-FALSE."""
    crc = 0xFFFF
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


@dataclass(frozen=True)
class Payload:
    """The one application value carried by a chirp."""

    take_id: int

    def __post_init__(self) -> None:
        if not 0 <= self.take_id < (1 << TAKE_ID_BITS):
            raise ValueError("take id out of range")

    @property
    def take(self) -> str:
        return take_id_to_str(self.take_id)


def encode_payload(payload: Payload) -> bytes:
    """Encode a take ID and its error-detecting checksum."""
    body = payload.take_id.to_bytes(TAKE_ID_BYTES, "big")
    return body + struct.pack(">H", crc16(body))


def decode_payload(data: bytes) -> Payload | None:
    """Decode a protected take ID, or return ``None`` if it is corrupt."""
    if len(data) != PAYLOAD_BYTES:
        return None
    body, tail = data[:TAKE_ID_BYTES], data[TAKE_ID_BYTES:]
    if crc16(body) != struct.unpack(">H", tail)[0]:
        return None
    return Payload(take_id=int.from_bytes(body, "big"))
