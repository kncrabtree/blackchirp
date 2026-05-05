# Bundle 04 — `BCLIF` and `BCLifTrace` Implementation

**Status:** not started
**Depends on:** 01, 02
**Blocks:** 05
**Effort:** L (3+ sessions)

## Scope

Replace the stub `BCLIF` class (currently just loads `lifparams.csv`
and `processing.csv` as DataFrames) with a full read-and-process
implementation that mirrors the routine LIF processing operations
available inside Blackchirp itself.

The C++ reference is `LifTrace` (`src/data/lif/liftrace.{h,cpp}`),
specifically the `LifProcSettings` struct, `processXY`, and
`integrate`. The on-disk schema is documented in
`doc/source/user_guide/lif/data_storage.rst`.

## Surface

### `BCLifTrace` — single scan-point trace

Analog of `BCFid` for LIF data. Constructed by `BCLIF.get_trace`; not
intended for direct user instantiation.

| Method                                            | Returns                              |
|---------------------------------------------------|--------------------------------------|
| `x()`                                             | `np.ndarray` (time, seconds)         |
| `lif()`                                           | `np.ndarray` (volts)                 |
| `ref()`                                           | `np.ndarray` or `None`               |
| `xy()`                                            | `(x, lif)` or `(x, lif, ref)` tuple  |
| `smooth(low_pass=None, savgol=None, **kwargs)`    | `np.ndarray` (smoothed lif)          |
| `integrate(lif_start=None, lif_end=None, ref_start=None, ref_end=None, **proc)` | `float` |

Smoothing parameters (`low_pass_alpha`, `savgol_window`,
`savgol_poly`) and integration gates default to the values loaded
from `lif/processing.csv`. Each can be overridden per call via
keyword argument, mirroring `BCFid.ft`.

`integrate` follows the C++ `LifTrace::integrate` semantics exactly:
trapezoidal sum in *sample-point* space (gates are sample indices,
not times), with `lif_integral / ref_integral` when a reference
channel is present, else raw `lif_integral`.

`smooth` implements the same two-stage chain as
`LifTrace::processXY`: IIR low-pass `y[i] = α·y[i-1] + (1-α)·y[i]`
followed by `scipy.signal.savgol_filter(window, polyorder)`.

### `BCLIF` — scan container

| Attribute                                | Description                                                    |
|------------------------------------------|----------------------------------------------------------------|
| `lifparams` (`pd.DataFrame`)             | Contents of `lif/lifparams.csv` (kept as DataFrame).           |
| `proc` (`dict`)                          | Contents of `lif/processing.csv` as `{key: value}`.            |
| `delay_points`, `delay_start`, `delay_step`, `delay_units` | Scan axis params from `header.csv`.   |
| `laser_points`, `laser_start`, `laser_step`, `laser_units` | Scan axis params from `header.csv`.   |
| `has_ref` (`bool`)                       | True if `refsize > 0` in any `lifparams` row.                  |
| `numtraces` (`int`)                      | `len(lifparams)`.                                              |

| Method                                                          | Returns                                |
|-----------------------------------------------------------------|----------------------------------------|
| `get_trace(l_index, d_index)`                                   | `BCLifTrace`                           |
| `delay_axis()`                                                  | `(np.ndarray, units_str)`              |
| `laser_axis()`                                                  | `(np.ndarray, units_str)`              |
| `delay_slice(l_index, fill=np.nan, **proc)`                     | `(delay_arr, integrals)`               |
| `laser_slice(d_index, fill=np.nan, **proc)`                     | `(laser_arr, integrals)`               |
| `image(fill=np.nan, **proc)`                                    | `(delay_arr, laser_arr, 2d_integrals)` |

**Missing-point handling.** `lifparams.csv` is sparse; the file index
`N = d_index * laser_points + l_index` may not exist. The aggregating
helpers (`image`, `delay_slice`, `laser_slice`) accept `fill=np.nan`
(default) or `fill=0.0` to choose how missing points are represented.
`get_trace` raises `KeyError` when the requested point is absent.

**Per-call processing override.** All aggregating helpers accept the
same processing kwargs as `BCLifTrace.integrate` (`lif_start`,
`lif_end`, `ref_start`, `ref_end`, `low_pass_alpha`, `savgol_window`,
`savgol_poly`, `savgol_enabled`); when omitted, defaults come from
`self.proc`.

## Header introspection

`delay_axis` and `laser_axis` are computed from `BCExperiment.header`
rows under `ObjKey == 'LifConfig'`:

- `DelayStart`, `DelayStep`, `DelayPoints`, units from the `Units`
  column for `DelayStart` (typically `μs`).
- `LaserStart`, `LaserStep`, `LaserPoints`, units from the `Units`
  column for `LaserStart` (typically `nm`).

`BCLIF` is constructed from `BCExperiment` and gets a reference to
the parent's header DataFrame, so it does not re-read the file. (Pass
`exp.header` as a constructor argument; do not introduce a circular
import.)

## Trace decoding

Mirrors `BCFid.__init__`:

- Read `lif/N.csv` with the experiment delimiter. Columns are `lif`
  (single-channel) or `lif;ref` (with reference).
- Convert each cell from base-36 string → int via
  `np.frompyfunc(int, 2, 1)`.
- Per-shot voltage: `raw / shots * ymult` (separately for `lif` and
  `ref` columns).

## Tests

Add under `python/blackchirp/tests/`, in the same broad-but-shallow
style as the FTMW tests in bundle 03 — the goal is to exercise every
coordination path, enum-handling path, and parameter combination,
asserting that well-formed inputs pass through without raising and
return values of the expected shape/type. Numerical accuracy of
`scipy.signal.savgol_filter` and the trapezoidal integral is taken on
faith.

- `test_lif_trace.py` — load a single trace from `v2-lif/`. Assert
  `x()` length matches `lifsize` and spacing matches the param row.
  Call `lif()`, `ref()`, `xy()` and assert returned shapes and
  dtypes. Exercise `smooth` with each combination of (`low_pass`
  only, `savgol` only, both, neither), asserting finite output.
  Exercise `integrate` with default gates, with override gates, and
  (for the with-ref fixture) confirm the ratio path is taken when
  ref gates are provided.
- `test_lif_scan.py` — exercise `delay_axis` / `laser_axis`
  (shape, dtype, units returned). Exercise `delay_slice` /
  `laser_slice` against a partial-scan fixture: confirm NaN appears
  where points are missing, then re-run with `fill=0` and confirm
  zero appears in the same positions. Exercise `image()` similarly.
  Verify shape is `(delay_points, laser_points)`.
- `test_lif_proc_overrides.py` — exercise the full kwarg surface on
  the aggregating helpers (`image`, `delay_slice`, `laser_slice`)
  one kwarg at a time, asserting no exception and finite output for
  the populated points. Cover `lif_start`, `lif_end`, `ref_start`,
  `ref_end`, `low_pass_alpha`, `savgol_window`, `savgol_poly`,
  `savgol_enabled`.
- `test_lif_missing_point.py` — request a `get_trace(l, d)` for an
  index known to be missing in the fixture; assert `KeyError` is
  raised. Request a present index and assert success.
- `test_lif_v1_absent.py` — load `mtbe/`, assert `exp.lif is None`
  or that the attribute is absent (whichever the existing
  `BCExperiment` does after bundle 03).
- `test_lif_no_ref.py` — if the fixture set includes a single-
  channel LIF capture (no ref), exercise the no-ref branch
  end-to-end. If only a with-ref fixture is available, monkey-patch
  `lifparams` to zero out `refsize` and confirm `ref()` returns
  `None` and `integrate` skips the ratio.

## Files touched

- `python/blackchirp/src/blackchirp/bclif.py` — full rewrite.
- `python/blackchirp/src/blackchirp/__init__.py` — re-export
  `BCLifTrace` alongside `BCLIF`.
- `python/blackchirp/src/blackchirp/blackchirpexperiment.py` — pass
  `self.header` into `BCLIF(...)` constructor.
- `python/blackchirp/tests/test_lif_trace.py`,
  `python/blackchirp/tests/test_lif_scan.py`,
  `python/blackchirp/tests/test_lif_v1_absent.py` — new.

## Acceptance criteria

- `BCExperiment('python/example-data/v2-lif')` loads with
  `exp.lif.numtraces` matching `len(lifparams.csv)`.
- `exp.lif.get_trace(l, d)` returns a `BCLifTrace` for a present
  index and raises `KeyError` for a missing one. `integrate()`
  returns a finite float for both default and override gates and
  takes the ratio path when ref gates are provided on a with-ref
  trace.
- `exp.lif.image()` returns a `(delay_points, laser_points)` array
  with NaN at indices not present in `lifparams`.
- `exp.lif.image(fill=0.0)` substitutes 0 at the same indices.
- Every kwarg on the aggregating helpers exercises without raising
  on the v2 fixture.
- v1 fixture: no regression, `BCExperiment` still has no `lif`
  attribute (or `lif is None`, per bundle 03).
- `pytest python/blackchirp/tests/` passes.
- `pylint -E` and `black --check` clean.

## Out of scope

- Peak finding (out of scope for the module per project convention).
- Plotting helpers.
- Multi-experiment LIF stitching or co-averaging.
- LIF data export to other formats.
