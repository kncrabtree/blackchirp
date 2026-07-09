"""Tests for the Sirah wire-protocol port (sirah_protocol.py)."""

from __future__ import annotations

import struct

from sirah_protocol import CMD_POSITION_QUERY, build_command, parse_status


def test_build_command_position_query_frame() -> None:
    """The no-arg 0x17 status query has zero-padded args and a 2-byte checksum."""
    frame = build_command(CMD_POSITION_QUERY)
    assert len(frame) == 13
    assert frame[0] == 0x3C
    assert frame[1] == CMD_POSITION_QUERY
    assert frame[12] == 0x3E
    assert frame[2:11] == bytes(9)
    assert frame[11] == (0x3C + CMD_POSITION_QUERY) & 0xFF


def test_build_command_checksum_wraps_mod_256() -> None:
    """The checksum is a running sum truncated to one byte, like the C++ char accumulator."""
    frame = build_command(0xFF, bytes([0xFF, 0xFF]))
    expected_checksum = (0x3C + 0xFF + 0xFF + 0xFF) & 0xFF
    assert frame[11] == expected_checksum


def test_build_command_truncates_long_args() -> None:
    """Only the first 9 argument bytes are used, matching the C++ loop bound."""
    frame = build_command(0x06, bytes(range(20)))
    assert frame[2:11] == bytes(range(9))


def _status_frame(
    err: int, c_status: int, m1_status: int, m1_pos: int, m2_status: int, m2_pos: int
) -> bytes:
    return (
        bytes([0x5B, err, c_status, m1_status])
        + struct.pack("<i", m1_pos)
        + bytes([m2_status])
        + struct.pack("<i", m2_pos)
        + bytes([0x5D])
    )


def test_parse_status_round_trip_positive_position() -> None:
    frame = _status_frame(
        err=0, c_status=1, m1_status=0, m1_pos=123456, m2_status=0, m2_pos=654321
    )
    status = parse_status(frame)
    assert status is not None
    assert status.err == 0
    assert status.c_status == 1
    assert status.m1_status == 0
    assert status.m1_pos == 123456
    assert status.m2_pos == 654321


def test_parse_status_round_trip_negative_position() -> None:
    """Negative motor positions round-trip through the little-endian signed decode."""
    frame = _status_frame(
        err=0, c_status=1, m1_status=1, m1_pos=-42, m2_status=0, m2_pos=-999999
    )
    status = parse_status(frame)
    assert status is not None
    assert status.m1_pos == -42
    assert status.m2_pos == -999999


def test_parse_status_rejects_wrong_length() -> None:
    assert parse_status(b"short") is None


def test_parse_status_rejects_bad_markers() -> None:
    frame = bytearray(_status_frame(0, 0, 0, 1, 0, 1))
    frame[0] = 0x00
    assert parse_status(bytes(frame)) is None
