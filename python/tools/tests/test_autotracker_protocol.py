"""Tests for the Autotracker wire-protocol port (autotracker_protocol.py)."""

from __future__ import annotations

from autotracker_protocol import (
    CMD_GOTO,
    CMD_POSITION_QUERY,
    RESPONSE_ID_POSITION,
    build_command,
    error_string,
    pack_pos24,
    parse_response,
    unpack_pos24,
)

# Bench golden vectors captured against a real FCU (§4.6 of the protocol
# reference / FCUCtrl.ipynb).
GOLDEN_GET_POSITION_CMD = bytes.fromhex("3E0017010000000000000056")
GOLDEN_GET_POSITION_RESP = bytes.fromhex("3C810B01FAE9BD0000000069")
GOLDEN_POSITION = 16443837


def test_build_command_matches_bench_get_position() -> None:
    """Get Position for motor 1 reproduces the bench command bytes."""
    frame = build_command(CMD_POSITION_QUERY, bytes([0x01]))
    assert frame == GOLDEN_GET_POSITION_CMD
    assert len(frame) == 12
    assert frame[0] == 0x3E
    assert frame[1] == 0x00
    assert frame[2] == CMD_POSITION_QUERY
    assert frame[11] == 0x56


def test_build_command_matches_bench_goto() -> None:
    """Goto Position (motor 1, Wait=0, Rel=0, 0xFAE9BD) matches the bench bytes."""
    data = bytes([0x01, 0x00, 0x00]) + pack_pos24(GOLDEN_POSITION)
    frame = build_command(CMD_GOTO, data)
    assert frame == bytes.fromhex("3E0022010000FAE9BD000001")


def test_build_command_checksum_wraps_mod_256() -> None:
    """The checksum is the 8-bit sum of bytes 0..10."""
    frame = build_command(0xFF, bytes([0xFF, 0xFF]))
    assert frame[11] == (0x3E + 0x00 + 0xFF + 0xFF + 0xFF) & 0xFF


def test_build_command_truncates_long_data() -> None:
    """Only the first 8 data bytes are used, matching the C++ loop bound."""
    frame = build_command(0x01, bytes(range(20)))
    assert frame[3:11] == bytes(range(8))


def test_parse_response_accepts_bench_get_position() -> None:
    """The bench Get Position reply decodes to the golden position."""
    response = parse_response(GOLDEN_GET_POSITION_RESP)
    assert response is not None
    assert response.adr_status == 0x81
    assert response.id == RESPONSE_ID_POSITION
    assert response.payload[0] == 0x01
    assert unpack_pos24(response.payload, 1) == GOLDEN_POSITION


def test_parse_response_rejects_bad_start_byte() -> None:
    frame = bytearray(GOLDEN_GET_POSITION_RESP)
    frame[0] = 0x3D
    assert parse_response(bytes(frame)) is None


def test_parse_response_rejects_wrong_length() -> None:
    assert parse_response(GOLDEN_GET_POSITION_RESP[:11]) is None
    assert parse_response(b"short") is None


def test_parse_response_rejects_bad_checksum() -> None:
    frame = bytearray(GOLDEN_GET_POSITION_RESP)
    frame[11] ^= 0xFF
    assert parse_response(bytes(frame)) is None


def test_pack_unpack_pos24_round_trip() -> None:
    for value in (0, 1, 0x7FFFFF, 0x800000, 0xFFFFFF, GOLDEN_POSITION):
        packed = pack_pos24(value)
        assert len(packed) == 3
        assert unpack_pos24(packed) == value


def test_unpack_pos24_bench_value() -> None:
    assert unpack_pos24(bytes.fromhex("FAE9BD")) == GOLDEN_POSITION


def test_unpack_pos24_short_buffer_returns_zero() -> None:
    assert unpack_pos24(bytes.fromhex("FAE9")) == 0
    assert unpack_pos24(b"") == 0
    assert unpack_pos24(bytes.fromhex("00FAE9BD"), 2) == 0


def test_pack_pos24_discards_high_bits() -> None:
    assert unpack_pos24(pack_pos24(0xFFFAE9BD)) == 0xFAE9BD


def test_error_string_known_and_unknown_codes() -> None:
    assert error_string(0) == "No Error"
    assert "Checksum" in error_string(2)
    assert "Stack Underflow" in error_string(7)
    assert "Motor Number" in error_string(14)
    assert "Serial Buffer Overflow" in error_string(32)
    assert "200" in error_string(200)
