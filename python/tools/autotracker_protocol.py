"""Sirah Autotracker (FCU) binary wire protocol: frames and parsing.

Line-for-line port of the ``BC::Autotracker`` layer in
``src/hardware/core/liflaser/autotrackerprotocol.h``/``.cpp``, the wire
format spoken by the Sirah Autotracker / Frequency Conversion Unit
(FCU). Keep this module byte-for-byte consistent with that C++ source
if the wire format changes.

The Autotracker is a separate Sirah product from the Cobra dye-laser
grating controller and does NOT share the Cobra's protocol -- the two
have only the broad shape of a framed command/response exchange with a
trailing checksum in common. See the C++ header's file comment for the
full command set and byte-layout detail.

Command frame (host -> Autotracker), 12 bytes, no terminator:
    byte 0  = 0x3E start marker
    byte 1  = 0x00 reserved
    byte 2  = command code
    bytes 3..10 = up to 8 data bytes (zero-padded)
    byte 11 = 8-bit checksum, sum(bytes[0:11]) & 0xFF

Response frame (Autotracker -> host), 12 bytes, no terminator:
    byte 0  = 0x3C start marker
    byte 1  = Adr/Status (bus address in low bits; MSB is NOT a
              reliable per-reply error flag -- see the C++ header)
    byte 2  = ID (self-describing payload tag)
    bytes 3..10 = 8 payload bytes
    byte 11 = checksum, same scheme as the command frame

Motor positions are 24-bit unsigned, transmitted big-endian (MSB
first).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

COMMAND_START_MARKER = 0x3E
RESPONSE_START_MARKER = 0x3C
COMMAND_LENGTH = 12
RESPONSE_LENGTH = 12

#: Get Position command byte; see ``SirahFcu::prompt()`` in sirahfcu.cpp.
CMD_POSITION_QUERY = 0x17
#: Goto Position command byte.
CMD_GOTO = 0x22
#: ID byte on a Get Position reply (``0x0b``).
RESPONSE_ID_POSITION = 0x0B

_ERROR_STRINGS = {
    0: "No Error",
    1: "Start Character error",
    2: "Checksum error",
    3: "Watchdog error",
    4: "Command Code error",
    5: "Command Break error",
    6: "Stack Overflow error",
    7: "Stack Underflow error",
    8: "AS Device error (bad device number)",
    9: "AS Inactive error (device inactive)",
    10: "AS Register error (bad register)",
    11: "AS LAM Source error",
    12: "AS LAM Code error",
    13: "AS Client error (device error)",
    14: "Motor Number error",
    15: "Motor Inactive error",
    16: "Motor Parameter error",
    17: "LED Parameter error",
    18: "Trace Parameter error",
    19: "Slope Parameter error",
    20: "Pre-Trigger Parameter error",
    21: "Pre-Trigger required error",
    22: "Trigger Level required error",
    23: "EXIO Parameter error",
    24: "Invert Parameter error",
    25: "Average Range error",
    26: "Timeout Occurred",
    27: "Timeout Parameter error",
    28: "Gain Parameter error",
    29: "Channel Parameter error",
    30: "Auto Timeout Parameter error",
    31: "External Trigger Parameter error",
    32: "Serial Buffer Overflow",
}


def build_command(cmd: int, data: bytes = b"") -> bytes:
    """Build a 12-byte Autotracker command frame.

    Byte 0 is the start marker (``0x3E``), byte 1 is the reserved zero
    byte, byte 2 is ``cmd``, bytes 3..10 hold up to 8 bytes of ``data``
    (zero-padded), and byte 11 is the 8-bit checksum of bytes 0..10.
    There is no terminator byte.

    Args:
        cmd: The command byte (placed at frame byte 2).
        data: Up to 8 bytes of command data; extra bytes are silently
            ignored, matching the C++ loop bound.

    Returns:
        The 12-byte command frame.
    """
    out = bytearray(COMMAND_LENGTH)
    out[0] = COMMAND_START_MARKER
    out[1] = 0x00
    out[2] = cmd & 0xFF
    for i, b in enumerate(data[:8]):
        out[3 + i] = b
    out[11] = sum(out[0:11]) & 0xFF
    return bytes(out)


@dataclass
class Response:
    """Parsed 12-byte Autotracker response frame.

    Attributes:
        adr_status: Byte 1 (Adr/Status). Bus address in the low bits;
            the MSB is NOT a reliable per-reply error flag (the bench
            shows it set on successful replies). Query the Error
            command for authoritative error state.
        id: Byte 2. Self-describing tag identifying the shape of
            ``payload``.
        payload: Bytes 3..10 (8 bytes); interpretation depends on
            ``id``.
    """

    adr_status: int
    id: int
    payload: bytes


def parse_response(resp: bytes) -> Optional[Response]:
    """Parse a 12-byte Autotracker response frame.

    Requires ``resp`` to be exactly 12 bytes, start with ``0x3C``, and
    carry a valid checksum in byte 11 (the 8-bit sum of bytes 0..10).
    Does not inspect the Adr/Status error-flag bit -- see the module
    docstring for why that bit is not a usable per-reply error
    indicator.

    Args:
        resp: The raw response bytes.

    Returns:
        The parsed :class:`Response`, or ``None`` if ``resp`` is not a
        well-formed 12-byte frame with a matching checksum.
    """
    if len(resp) != RESPONSE_LENGTH or resp[0] != RESPONSE_START_MARKER:
        return None
    if (sum(resp[0:11]) & 0xFF) != resp[11]:
        return None

    return Response(adr_status=resp[1], id=resp[2], payload=bytes(resp[3:11]))


def pack_pos24(pos: int) -> bytes:
    """Pack a position as a 3-byte, big-endian (MSB-first) 24-bit field.

    The low 24 bits of ``pos`` are used; any higher bits are discarded.

    Args:
        pos: The position in steps.

    Returns:
        Three bytes, MSB first.
    """
    value = pos & 0xFFFFFF
    return bytes([(value >> 16) & 0xFF, (value >> 8) & 0xFF, value & 0xFF])


def unpack_pos24(b: bytes, off: int = 0) -> int:
    """Unpack a 3-byte, big-endian (MSB-first) 24-bit position field.

    Args:
        b: The buffer to read from.
        off: The offset of the field's first (most significant) byte.

    Returns:
        The decoded unsigned value, or 0 if ``b`` does not hold at
        least 3 bytes starting at ``off``.
    """
    if off < 0 or len(b) < off + 3:
        return 0
    return (b[off] << 16) | (b[off + 1] << 8) | b[off + 2]


def error_string(code: int) -> str:
    """Human-readable message for an Autotracker error code.

    Mirrors ``BC::Autotracker::errorString()`` for the documented
    codes 0..32.

    Args:
        code: The numeric error code.

    Returns:
        A descriptive message, including the numeric code for any code
        outside the documented 0..32 range.
    """
    if code in _ERROR_STRINGS:
        return _ERROR_STRINGS[code]
    return f"Unknown Autotracker error code {code}"
