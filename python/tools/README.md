# Sirah FCU calibration tools

Standalone operator scripts for the offline Sirah FCU (frequency-doubling
unit) calibration workflow described in
`dev-docs/sirah-fcu-calibration.md`. These are **not** part of the
importable `blackchirp` package (`python/blackchirp/`) and are not
installed by it — they live here so the package's minimal-dependency
contract (numpy/scipy/pandas only; see `python/AGENTS.md`) is unaffected.

- `fcu_measure.py` — interactive tool: prompts for a fundamental
  wavelength, reads the FCU's motor position over serial (or synthesizes
  one with `--simulate`), and appends `wavelengthNm;positionSteps` rows
  to a measurements CSV.
- `fcu_fit.py` — reads a measurements CSV and fits/visualizes the tuning
  curve under all three of Blackchirp's `FcuCalibration` schemes
  (Physical / Polynomial / Spline), printing the Physical scheme's five
  parameters and exporting Polynomial/Spline import CSVs.
- `fcu_calibration_math.py` — the Physical scheme's Sellmeier + phase-match
  + sine-bar math, ported line-for-line from `src/data/lif/fcucalibration.cpp`
  so fitted parameters reproduce exactly when typed into Blackchirp.
- `autotracker_protocol.py` — the Sirah Autotracker (FCU) binary wire
  protocol (command framing, response parsing, 24-bit position
  pack/unpack, error codes), ported from
  `src/hardware/core/liflaser/autotrackerprotocol.h`/`.cpp`.
- `fcu_csv.py` — shared readers/writers for the three semicolon-delimited
  CSV shapes in play (measurements, spline import, polynomial import).

## Dependencies

Beyond the standard library:

- `fcu_fit.py` needs **numpy**, **scipy**, and **matplotlib**. All three
  are already in the `blackchirp-py` conda environment
  (`python/environment.yml`) for the example notebooks.
- `fcu_measure.py` needs **pyserial** for hardware mode only; `--simulate`
  mode needs nothing beyond the standard library. `pyserial` is **not**
  currently in `python/environment.yml` / `python/requirements.txt` —
  install it into your environment before recording real measurements
  (`conda install pyserial` / `pip install pyserial`).

Neither dependency is added to `python/blackchirp/pyproject.toml`; that
file's dependency list is the installed package's contract and is
deliberately left alone by these tools.

## Usage

```bash
# Record measurements on the bench (Ctrl+D or a blank line to finish):
python fcu_measure.py --port /dev/ttyUSB1 --output measurements.csv

# Or without hardware:
python fcu_measure.py --simulate --output measurements.csv

# Fit and visualize:
python fcu_fit.py measurements.csv --crystal bbo --cut-angle 32.3 \
    --save calibration.png
```

`fcu_fit.py --help` / `fcu_measure.py --help` document the full CLI.

## Tests

Pure-function-level pytest coverage lives under `tests/`. Run with the
`blackchirp-py` conda environment (same one used for the `blackchirp`
package's own suite):

```bash
conda run -n blackchirp-py python -m pytest python/tools/tests
```

These tests only exercise `--simulate` mode, so they do not require
`pyserial`.
