#!/usr/bin/env python
"""Interactive measurement-logging tool for a Sirah FCU calibration run.

Repeatedly prompts the operator for a fundamental wavelength (nm;
blank input finishes the session), hand-tunes are done externally
(the operator peaks the doubled output at the bench, per the Sirah
Autotracker service manual's calibration procedure), then reads the
FCU's current motor position over serial and appends
``wavelengthNm;positionSteps`` to the output CSV. The output feeds
``fcu_fit.py`` directly.

Speaks the Sirah Autotracker binary wire protocol (see
:mod:`autotracker_protocol`): sends the framed Get Position command
(``0x17``, with the motor number as data byte 0) and decodes the
24-bit big-endian position from the 12-byte reply.

Example:
    python fcu_measure.py --port /dev/ttyUSB1 --output measurements.csv

    python fcu_measure.py --simulate --output measurements.csv
"""

from __future__ import annotations

import argparse
import sys
from typing import Callable, Optional, Sequence

from fcu_csv import append_measurement_csv
from autotracker_protocol import (
    CMD_POSITION_QUERY,
    RESPONSE_ID_POSITION,
    RESPONSE_LENGTH,
    build_command,
    parse_response,
    unpack_pos24,
)

try:
    import serial  # pyserial; optional so --simulate works without it installed.
except ImportError:  # pragma: no cover - exercised only when pyserial is absent
    serial = None


def read_position_hardware(ser: "serial.Serial", motor: int = 1) -> int:
    """Query the FCU's current motor position over an open serial port.

    Sends Get Position (``0x17``) with the motor number as data byte 0,
    reads the 12-byte reply, verifies the ID byte (``0x0b``) and the
    echoed motor number, then decodes the 24-bit big-endian position
    from reply bytes 4..6.

    Args:
        ser: An open ``pyserial`` ``Serial`` instance connected to the
            FCU's RS232 port.
        motor: The Autotracker motor index (1..3) to query.

    Returns:
        The current motor position in steps.

    Raises:
        RuntimeError: If the response is not a well-formed Get Position
            frame, the ID byte is wrong, or the echoed motor number
            does not match the request.
    """
    ser.reset_input_buffer()
    ser.write(build_command(CMD_POSITION_QUERY, bytes([motor & 0xFF])))
    resp = ser.read(RESPONSE_LENGTH)
    response = parse_response(resp)
    if response is None or response.id != RESPONSE_ID_POSITION:
        raise RuntimeError(f"Unexpected response from FCU (hex: {resp.hex()})")
    if not response.payload or response.payload[0] != (motor & 0xFF):
        echoed = response.payload[0] if response.payload else None
        raise RuntimeError(
            f"Get Position echoed motor {echoed}, expected {motor} (hex: {resp.hex()})"
        )
    # Payload byte 0 is the echoed motor number; the 24-bit position sits
    # at payload bytes 1..3 (i.e. reply bytes 4..6).
    return unpack_pos24(response.payload, 1)


def simulated_position(wavelength_nm: float) -> int:
    """A deterministic, monotone-decreasing synthetic motor position.

    Not a physical model of any crystal — just a repeatable stand-in
    so ``--simulate`` is exercisable (and unit-testable) without
    hardware or a fitted calibration.

    Args:
        wavelength_nm: The fundamental wavelength the operator entered.

    Returns:
        A synthetic motor position in steps.
    """
    return int(round(2_000_000.0 - 4000.0 * wavelength_nm))


def run_session(
    get_position: Callable[[float], int],
    output_path: str,
    input_func: Callable[[str], str] = input,
    print_func: Callable[[str], None] = print,
) -> int:
    """Run the interactive prompt/record loop.

    Args:
        get_position: Called with the entered wavelength (nm); returns
            the motor position in steps to record.
        output_path: Measurements CSV to append to.
        input_func: Injectable stand-in for :func:`input` (testing).
        print_func: Injectable stand-in for :func:`print` (testing).

    Returns:
        The number of measurement points recorded.
    """
    recorded = 0
    while True:
        raw = input_func("Fundamental wavelength (nm), blank to finish: ").strip()
        if raw == "":
            break
        try:
            wavelength_nm = float(raw)
        except ValueError:
            print_func(f"Could not parse {raw!r} as a wavelength in nm; try again.")
            continue

        try:
            position_steps = get_position(wavelength_nm)
        except Exception as exc:  # noqa: BLE001 - report and keep the session alive
            print_func(f"Failed to read position: {exc}")
            continue

        append_measurement_csv(output_path, wavelength_nm, position_steps)
        recorded += 1
        print_func(
            f"Recorded {wavelength_nm:g} nm -> {position_steps} steps (appended to {output_path})"
        )

    print_func(f"Done: {recorded} point(s) written to {output_path}")
    return recorded


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Interactively record wavelengthNm;positionSteps calibration points for a "
            "Sirah FCU, by querying its motor position over serial after each hand-tune."
        )
    )
    parser.add_argument(
        "--port",
        default=None,
        help="Serial device (e.g. /dev/ttyUSB0). Required unless --simulate.",
    )
    parser.add_argument(
        "--baud",
        type=int,
        default=19200,
        help="Serial baud rate (default: 19200, the Autotracker's fixed serial rate).",
    )
    parser.add_argument(
        "--motor",
        type=int,
        default=1,
        choices=(1, 2, 3),
        help="Autotracker motor index (1-3) driving the doubling crystal (default: 1).",
    )
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=200,
        help="Serial read timeout in milliseconds (default: 200, the FCU's registered comm timeout).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output measurements CSV (wavelengthNm;positionSteps); created with a header if "
        "new, appended to otherwise.",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Return a synthetic monotone position instead of opening a serial port, so the "
        "tool is exercisable without hardware.",
    )
    return parser


def _open_hardware_reader(
    args: argparse.Namespace,
) -> "tuple[Callable[[float], int], serial.Serial]":
    if serial is None:
        raise RuntimeError(
            "pyserial is required for hardware mode (pip/conda install pyserial), or pass --simulate."
        )
    ser = serial.Serial(args.port, args.baud, timeout=args.timeout_ms / 1000.0)

    def _read(_wavelength_nm: float) -> int:
        return read_position_hardware(ser, args.motor)

    return _read, ser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point: parse args, open the position source, and run the session."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.simulate:
        get_position: Callable[[float], int] = simulated_position
        ser = None
    else:
        if not args.port:
            parser.error("--port is required unless --simulate is given.")
        get_position, ser = _open_hardware_reader(args)

    try:
        run_session(get_position, args.output)
    finally:
        if ser is not None:
            ser.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
