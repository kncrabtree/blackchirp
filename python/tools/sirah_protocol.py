"""Sirah binary wire protocol: frame construction and status parsing.

Line-for-line port of ``BC::Sirah::buildCommand()`` /
``BC::Sirah::parseStatus()`` in
``src/hardware/core/liflaser/sirahprotocol.h``/``.cpp``, the wire
format shared by the Sirah Cobra grating and FCU frequency-conversion
drivers. Keep this module consistent with that C++ source if the wire
format changes.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Optional

START_MARKER = 0x3C
END_MARKER = 0x3E
STATUS_START_MARKER = 0x5B
STATUS_END_MARKER = 0x5D
COMMAND_LENGTH = 13
STATUS_LENGTH = 14

#: Position-query command byte; see ``SirahFcu::prompt()`` in sirahfcu.cpp.
CMD_POSITION_QUERY = 0x17


def build_command(cmd: int, args: bytes = b"") -> bytes:
    """Build a 13-byte binary command frame.

    Byte 0 is the start marker (``0x3c``), byte 1 is ``cmd``, bytes
    2..10 hold up to 9 bytes of ``args`` (zero-padded), byte 11 is a
    running one-byte checksum (mod 256) accumulated over the start
    marker, command, and args bytes, and byte 12 is the end marker
    (``0x3e``).

    Args:
        cmd: The command byte.
        args: Up to 9 bytes of command arguments; extra bytes are
            silently ignored, matching the C++ loop bound.

    Returns:
        The 13-byte command frame.
    """
    out = bytearray(COMMAND_LENGTH)
    out[0] = START_MARKER
    out[1] = cmd & 0xFF
    checksum = (out[0] + out[1]) & 0xFF
    for i, b in enumerate(args[:9]):
        out[2 + i] = b
        checksum = (checksum + b) & 0xFF
    out[11] = checksum
    out[12] = END_MARKER
    return bytes(out)


@dataclass
class Status:
    """Parsed 14-byte status response; mirrors ``BC::Sirah::Status``.

    Attributes:
        err: Device error code.
        c_status: Controller status byte.
        m1_status: Motor-1 status byte (bit 0: motor running).
        m1_pos: Motor-1 position, signed 32-bit steps.
        m2_status: Motor-2 status byte.
        m2_pos: Motor-2 position, signed 32-bit steps.
    """

    err: int
    c_status: int
    m1_status: int
    m1_pos: int
    m2_status: int
    m2_pos: int


def parse_status(resp: bytes) -> Optional[Status]:
    """Parse a 14-byte status response.

    Mirrors ``BC::Sirah::parseStatus()``, including the little-endian
    signed 32-bit motor-position decode (bytes 4..7 for ``m1_pos``,
    bytes 9..12 for ``m2_pos``).

    Args:
        resp: The raw response bytes.

    Returns:
        The parsed :class:`Status`, or ``None`` if ``resp`` is not a
        well-formed 14-byte frame bracketed by the expected start
        (``0x5b``) / end (``0x5d``) markers.
    """
    if (
        len(resp) != STATUS_LENGTH
        or resp[0] != STATUS_START_MARKER
        or resp[-1] != STATUS_END_MARKER
    ):
        return None

    (m1_pos,) = struct.unpack_from("<i", resp, 4)
    (m2_pos,) = struct.unpack_from("<i", resp, 9)
    return Status(
        err=resp[1],
        c_status=resp[2],
        m1_status=resp[3],
        m1_pos=m1_pos,
        m2_status=resp[8],
        m2_pos=m2_pos,
    )
