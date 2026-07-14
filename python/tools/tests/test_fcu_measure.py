"""Tests for the measurement-logging tool (fcu_measure.py).

Exercises the ``--simulate`` code path only — no serial hardware is
touched. The end-to-end test chains simulated measurement recording
into ``fcu_fit.py``'s fit/export, matching the plan's "measure ->
fit -> export runs end to end" requirement.
"""

from __future__ import annotations

from typing import Iterator, List

import pytest

from autotracker_protocol import CMD_POSITION_QUERY, build_command
from fcu_csv import read_measurements_csv
from fcu_measure import (
    build_arg_parser,
    read_position_hardware,
    run_session,
    simulated_position,
)


def _fake_input(responses: Iterator[str]):
    def _input(_prompt: str) -> str:
        return next(responses)

    return _input


class _FakeSerial:
    """Minimal stand-in for a pyserial ``Serial`` for read/write assertions."""

    def __init__(self, reply: bytes) -> None:
        self._reply = reply
        self.written = b""
        self.reset_count = 0

    def reset_input_buffer(self) -> None:
        self.reset_count += 1

    def write(self, data: bytes) -> int:
        self.written = bytes(data)
        return len(data)

    def read(self, n: int) -> bytes:
        return self._reply[:n]


def test_read_position_hardware_decodes_bench_reply() -> None:
    """The bench Get Position reply decodes to the golden 24-bit position."""
    reply = bytes.fromhex("3C810B01FAE9BD0000000069")
    ser = _FakeSerial(reply)

    position = read_position_hardware(ser, motor=1)

    assert position == 16443837
    assert ser.reset_count == 1
    # Command carries the motor number as data byte 0 (bench: 3E 00 17 01 ...).
    assert ser.written == build_command(CMD_POSITION_QUERY, bytes([0x01]))


def test_read_position_hardware_rejects_wrong_motor_echo() -> None:
    """A reply echoing a different motor number is a hard error."""
    # Same golden reply (motor 1) queried while asking for motor 2.
    ser = _FakeSerial(bytes.fromhex("3C810B01FAE9BD0000000069"))
    with pytest.raises(RuntimeError):
        read_position_hardware(ser, motor=2)


def test_read_position_hardware_rejects_bad_frame() -> None:
    """A malformed frame (bad checksum) raises rather than returning garbage."""
    ser = _FakeSerial(bytes.fromhex("3C810B01FAE9BD00000000FF"))
    with pytest.raises(RuntimeError):
        read_position_hardware(ser, motor=1)


def test_cli_defaults_baud_and_motor() -> None:
    """The CLI defaults to 19200 baud and motor 1."""
    parser = build_arg_parser()
    args = parser.parse_args(["--output", "out.csv", "--simulate"])
    assert args.baud == 19200
    assert args.motor == 1


def test_simulated_position_is_monotone_and_deterministic() -> None:
    wavelengths = [600.0, 700.0, 800.0, 900.0]
    positions = [simulated_position(wl) for wl in wavelengths]
    assert positions == sorted(positions, reverse=True)
    assert len(set(positions)) == len(positions)
    assert [simulated_position(wl) for wl in wavelengths] == positions  # deterministic


def test_run_session_records_valid_entries_and_stops_on_blank(tmp_path) -> None:
    csv_path = tmp_path / "measurements.csv"
    responses = iter(["700", "750.5", ""])
    messages: List[str] = []

    recorded = run_session(
        simulated_position,
        str(csv_path),
        input_func=_fake_input(responses),
        print_func=messages.append,
    )

    assert recorded == 2
    wavelengths_nm, positions_steps = read_measurements_csv(str(csv_path))
    assert wavelengths_nm == [700.0, 750.5]
    assert positions_steps == [simulated_position(700.0), simulated_position(750.5)]
    assert any("Done: 2 point" in m for m in messages)


def test_run_session_tolerates_unparsable_input(tmp_path) -> None:
    csv_path = tmp_path / "measurements.csv"
    responses = iter(["not-a-number", "700", ""])
    messages: List[str] = []

    recorded = run_session(
        simulated_position,
        str(csv_path),
        input_func=_fake_input(responses),
        print_func=messages.append,
    )

    assert recorded == 1
    wavelengths_nm, _positions_steps = read_measurements_csv(str(csv_path))
    assert wavelengths_nm == [700.0]
    assert any("Could not parse" in m for m in messages)


def test_simulate_measure_then_fit_export_end_to_end(tmp_path) -> None:
    """--simulate measurement session -> fcu_fit.main() fit/export, no crash."""
    import fcu_fit

    measurements_csv = tmp_path / "measurements.csv"
    wavelengths = [650.0, 700.0, 750.0, 800.0, 850.0, 900.0]
    responses = iter([str(w) for w in wavelengths] + [""])

    recorded = run_session(
        simulated_position,
        str(measurements_csv),
        input_func=_fake_input(responses),
        print_func=lambda *_a: None,
    )
    assert recorded == len(wavelengths)

    poly_output = tmp_path / "poly.csv"
    spline_output = tmp_path / "spline.csv"
    exit_code = fcu_fit.main(
        [
            str(measurements_csv),
            "--crystal",
            "bbo",
            "--cut-angle",
            "32.3",
            "--poly-degree",
            "3",
            "--poly-output",
            str(poly_output),
            "--spline-output",
            str(spline_output),
            "--no-plot",
        ]
    )

    assert exit_code == 0
    poly_lines = poly_output.read_text(encoding="utf-8").splitlines()
    assert poly_lines[0] == "order;forward;inverse"
    assert len(poly_lines) == 1 + 4  # header + orders 0..3

    spline_lines = spline_output.read_text(encoding="utf-8").splitlines()
    assert spline_lines[0] == "wavelengthNm;positionSteps"
    assert len(spline_lines) == 1 + len(wavelengths)
