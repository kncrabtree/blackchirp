# Blackchirp Python Module — Example Data

This directory holds small experiment fixtures that the Python module's
notebooks, tests, and docstrings exercise. Each fixture is a complete
on-disk experiment tree; the Blackchirp companion module reads it via
`BCExperiment("path/to/fixture")`.

## Fixtures

### `mtbe/`

Methyl tert-butyl ether single-FID acquisition captured with
**Blackchirp 1.0**. Numeric hardware identifiers (e.g. `Ftmw1`,
`PulseGen1`) and numeric enum cells. Retained as the v1 backward-compat
smoke test: any change that breaks loading of this fixture is a
regression in the module's older-format support.

### `v2-ftmw/`

Single-FID Forever acquisition captured with **Blackchirp 2.0 (devel)**.
Schema features exercised:

- Label-based hardware identifiers (e.g. `FlowController.Default`,
  `Ftmw.Default`).
- `markers.csv` populated with the four-channel default marker set
  (Protection, Gate, two custom).
- `processing.csv` with `BlackmanHarris` window function name and
  string-form `FtUnits;FtuV` (mapped back to an exponent by the
  refreshed module's processing path).

### `v2-lif-ref/`

Partial 6×3 (laser × delay) LIF scan captured with **Blackchirp 2.0
(devel)**, reference channel enabled. The scan stopped before the laser
axis completed, so several `(lIndex,dIndex)` cells are absent — useful
for exercising the missing-scan-point handling that `BCLIF.image()`
will need. `lifparams.csv` carries non-zero `refsize` and `refymult`
columns; per-point CSVs include both LIF and reference traces.

### `v2-lif-noref/`

Partial LIF scan captured with **Blackchirp 2.0 (devel)**, reference
channel disabled (`refsize;0`, `refymult;0`). Mirrors `v2-lif-ref/`
except for the absence of reference data, so `BCLIF` reference-channel
code paths can be exercised in both states without running a full v2
acquisition.

## Notes for downstream work

The current Python module is pre-refresh and has two known gaps that
the v2 fixtures expose:

1. `BCExperiment.__init__` unconditionally reads `clocks.csv`, which is
   not produced for LIF-only experiments. Loading either v2 LIF fixture
   raises `FileNotFoundError`. Bundle 03 makes the clocks read
   conditional on FTMW being present.
2. `BCFid.ft()` parses `FtUnits` as an integer exponent, but Blackchirp
   2.0 writes it as the enum name (`FtuV`, `FtmV`, …). Bundle 03 maps
   the name back to the exponent.

`v2-ftmw/` loads under the un-refreshed module today, but `fid.ft()`
will fail on the `FtUnits` string until bundle 03 lands. `mtbe/`
continues to load and process unchanged.
