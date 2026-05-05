# Bundle 03 — Python Module Schema Fixes (FTMW + Experiment Top Level)

**Status:** in progress
**Depends on:** 01, 02
**Blocks:** 05
**Effort:** M (2 sessions)

## Status notes

- 2026-05-05: Bundle opened. Two scope additions agreed with the
  maintainer before implementation began:
  1. **Differential FID API.** Mirror Blackchirp's "differential
     backup" view in the Python module. New `BCFTMW.get_differential_fid`
     returns shots collected between two backup points; details in
     §9 below.
  2. **Error-handling sweep.** Convert silent-empty-return paths
     ("return `""` on missing key") to explicit exceptions, since
     they confuse Python users. Details in §10 below. The two
     intentional exceptions (paths that keep their sentinel return)
     are listed there.
- No blockers; proceeding with implementation in the order listed
  in the Changes section.

## Scope

Bring `BCExperiment`, `BCFTMW`, and `BCFid` up to date with the
current Blackchirp on-disk schema. This bundle addresses every gap
identified in the planning analysis *except* LIF (covered in bundle
04) and documentation (covered in bundle 05).

The principle throughout: the module must read v1 fixtures (e.g.
`mtbe/`) and v2 fixtures (`v2-ftmw/`) with the same code path,
producing equivalent in-memory structures.

## Changes

### 1. `markers.csv` optional read on `BCExperiment`

Mirror the existing `chirps`/`auxdata` optional-read pattern in
`blackchirpexperiment.py`:

- Attempt to read `markers.csv` after `chirps.csv`.
- Store as `self.markers` (a `pd.DataFrame` with columns `Channel`,
  `Name`, `Role`, `TimingMode`, `StartUs`, `EndUs`, `Enabled`).
- Swallow `FileNotFoundError` only (markers.csv is absent when no
  marker-capable AWG is configured).
- Add `markers` to the docstring `Attributes` block.

### 2. Sideband string handling in `BCFTMW.process_sideband`

`bcftmw.py:214` checks `fid.fidparams["sideband"] == 1` and silently
falls into the `else` branch on current data, where `sideband` is the
string `"LowerSideband"`. Replace with the same dual-form check
already used in `BCFid.apply_lo`:

```python
if fid.fidparams["sideband"] == 1 or "Lower" in str(fid.fidparams["sideband"]):
```

Better: hoist this into a single helper on `BCFid` (e.g.
`is_lower_sideband() -> bool`) so both call sites stay in sync.

### 3. Robust window-function dispatch in `BCFid.ft`

`bcfid.py:269–289` has structural bugs (mixed `if`/`elif`,
substring matches that can collide). Replace with a name → scipy
window mapping:

```python
_WINDOW_MAP = {
    "None":           "boxcar",
    "Bartlett":       "bartlett",
    "Blackman":       "blackman",
    "BlackmanHarris": "blackmanharris",
    "Hamming":        "hamming",
    "Hanning":        "hann",
    "KaiserBessel":   ("kaiser", 14.0),
}
_WINDOW_INT_MAP = {0: "None", 1: "Bartlett", 2: "Blackman",
                   3: "BlackmanHarris", 4: "Hamming",
                   5: "Hanning", 6: "KaiserBessel"}
```

Lookup chain: explicit `winf` kwarg wins; else read `proc` value;
else default to `"None"`. If the value is an int (or int-shaped
string), translate via `_WINDOW_INT_MAP` first, then look up in
`_WINDOW_MAP`. Unknown values raise `ValueError` — silent fallback to
boxcar today is unhelpful.

### 4. Enum-or-int normalization helper

Add a small private helper in `bcfid.py` (or a new
`_enum_helpers.py` if more than one class needs it):

```python
def _resolve_enum(value, name_map: dict, *, default=None):
    """Map a CSV-cell value (int, int-shaped string, or enum name) to
    a canonical name. Returns ``default`` if the value matches
    nothing.
    """
```

Use this for sideband, window function, and any future enum field
the module reads. The helper centralizes the "either form" logic so
that bundle 04 (LIF) and any future schema field do not duplicate it.

### 5. `FtUnits` migration

After bundle 01 lands, `processing.csv` will contain
`FtUnits;FtuV` (or similar) instead of `FtUnits;6`. Add a
name → exponent map (`FtV → 0`, `FtmV → 3`, `FtuV → 6`, `FtnV → 9`)
and route the lookup through the helper from #4. The integer form
must continue to parse for `mtbe/` and any other v1 fixture.

### 6. `hardware.csv` reader hygiene

After bundle 01 lands, the `hardware.csv` second column is named
`driver` (formerly `subKey`), and the third `hardwareType` column may
be dropped, kept numeric, or promoted to a string-form enum
depending on the decision recorded in bundle 01's status header.
The Python reader must:

- Accept either header label (`subKey` or `driver`) on input so that
  v1 and v2 fixtures both load. Implement by normalizing the column
  name on read (e.g. rename `subKey` → `driver` after `read_csv`).
- Accept the `hardwareType` column as present-and-numeric,
  present-and-string, or absent, without raising.
- Optionally expose a small accessor that returns the per-row driver
  identity in canonical form. The DataFrame itself remains the
  public interface; the accessor is a convenience.

### 7. Docstring updates

- Update `BCExperiment` docstring example showing
  `exp.hardware` to use current label-based keys
  (`AWG.Ka`, `Clock.virtual`, `FtmwScope.virtual`, etc.) and the
  three-column layout.
- Update `BCExperiment` `Attributes` block to add `markers`.
- Trim or remove any references to the v1 numeric-only enum forms.

### 9. Differential FID API

Mirror the C++ `FidSingleStorage::loadDifferentialFidList` feature in
the Python module, with a more general two-bound interface:

```python
class BCFTMW:
    def is_multi_segment(self) -> bool: ...
    def num_backups(self) -> int: ...
    def get_differential_fid(self, start: int = 0, end: int = -1) -> BCFid: ...
```

Semantics (matches the C++ on-disk convention):

- `0.csv` is the cumulative final FID. `1.csv`, `2.csv`, … are
  backups taken at intermediate times; each contains the cumulative
  FID up through its backup point.
- `start=0` → no subtraction; the differential begins at the start
  of the experiment.
- `start=k` (`k > 0`) → subtract FID from `k.csv`.
- `end=-1` → use `0.csv` as the upper bound (i.e. "current/final").
- `end=k` (`k > 0`) → use `k.csv` as the upper bound.
- The returned `BCFid` reports
  `fidparams.shots = shots(end) − shots(start)` so that downstream
  scaling (`vmult / shots`) is correct.

`is_multi_segment` reads `FtmwType` from `objectives.csv` and returns
`True` iff the value is in `{LO_Scan, DR_Scan, Peak_Up}`. To support
this, `BCExperiment.__init__` must capture `FtmwType` from objectives
and pass it (or a flag) into the `BCFTMW` constructor — `BCFTMW`
itself does not read `objectives.csv`.

`num_backups()` returns `len(fidparams) − 1` for single-segment
experiments and `0` for multi-segment.

`get_differential_fid` must raise:

- `ValueError` if `is_multi_segment()` is true (with a message
  identifying the FtmwType).
- `ValueError` if `start < 0`, if `start > num_backups()`,
  if `end != -1` and not in `[1, num_backups()]`, or if the
  resolved `(start, end)` ordering does not satisfy `start < end`
  (with `end=-1` resolved to "after every backup").

Implementation: instantiate two `BCFid`s (start and end), subtract
the raw integer arrays element-wise, recompute `data` as
`_rawdata * vmult / shots_diff`, and patch the result's
`fidparams` row before returning. Frames must match between the two
inputs; if they do not (defensive check, should never trigger in
single-segment data), raise `ValueError`.

### 10. Error-handling sweep

Replace silent-empty-return paths with explicit exceptions. Update
each affected method's `Raises:` block to list every exception that
this module's own code raises (downstream pandas/numpy/scipy errors
are not enumerated unless the wrapper translates them).

Conversions:

- `BCExperiment.header_value(objKey, valKey, idx, arrKey)` — current
  behavior returns `""` when no row matches. Change to raise
  `KeyError` with a message naming the requested
  `(ObjKey, ValueKey, ArrKey, idx)`. The message should make the
  failed lookup unambiguous.
- `BCExperiment.header_unit(...)` — same conversion for the
  missing-row case. The "row exists, unit cell is empty/NaN" case
  remains a return of `""` (this is a legitimate "no units"
  answer). Document both behaviors in the docstring.
- `BCFid.ft` — wired through `_resolve_enum`; unknown window
  function name, unknown `FtUnits` value, and unknown sideband
  value raise `ValueError`. Today's silent fallback to `boxcar`
  / `units_power=0` is removed.
- `BCFTMW.get_fid` — already raises `ValueError`; keep as is, but
  ensure the `Raises:` block documents it.
- `BCFTMW.process_sideband` — already raises `ValueError` on bad
  `which` / `avg`; keep, document the full set in `Raises:`.
- `BCFTMW.get_differential_fid` — raises per §9.
- `BCExperiment.__init__` — already raises `FileNotFoundError`;
  add a `KeyError`/`ValueError` if `header.csv` lacks the
  `Experiment / Number` row that the constructor relies on
  (currently fails with an opaque `IndexError`).

Intentionally **not** converted (kept as sentinel returns):

- `BCExperiment.header_rows(...)` returns an empty DataFrame on no
  match. Empty DataFrame is the idiomatic pandas "zero matches"
  result and callers can use `.empty` / `len()` to branch. Document
  this explicitly in the docstring.
- `BCExperiment.header_unit(...)` — empty unit cell (NaN) returns
  `""` (see above; this case is "row found, no unit").

### 8. Tests

The Python module currently has no test suite. This bundle adds a
broad-but-shallow one under `python/blackchirp/tests/`. The intent
is **not** to verify basic pandas processing or numerical accuracy
of FT internals — those are covered by scipy upstream and by visual
inspection in the example notebooks. The intent **is** to exercise
all coordination paths, enum-handling paths, and parameter
combinations, asserting only that well-formed inputs pass through
without raising.

For FTMW coverage:

- `test_load_v1.py` — load `python/example-data/mtbe/`, smoke-test
  every public method on `BCExperiment` / `BCFTMW` / `BCFid` once
  with default arguments. Include `header_rows`, `header_value`,
  `header_unit`, `header_unique_keys`. Does not assert numeric
  values; asserts that calls return the expected types and shapes.
- `test_load_v2.py` — same coverage against
  `python/example-data/v2-ftmw/`. Additionally:
  - confirm `markers` is populated when the fixture has it (skip if
    not),
  - confirm the v2 `hardware.csv` `driver` column is read whether
    the header label is `subKey` or `driver` (parametrize on a
    monkey-patched fixture that flips the label).
- `test_window_functions.py` — for each window-function name
  (`None`, `Bartlett`, `Blackman`, `BlackmanHarris`, `Hamming`,
  `Hanning`, `KaiserBessel`) AND each numeric form (0–6), call
  `BCFid.ft(winf=...)` against a fixture FID and assert no exception
  is raised, the returned array is finite, and shape is as expected.
- `test_ft_units.py` — for each `FtUnits` value (both the string
  forms `FtV` / `FtmV` / `FtuV` / `FtnV` and the numeric forms 0 /
  3 / 6 / 9), call `BCFid.ft(units_power=...)` and assert the output
  scales as expected (intensity-ratio check between two settings is
  enough; no absolute-value assertion needed).
- `test_sideband_dispatch.py` — for each sideband form (`0`, `1`,
  `'LowerSideband'`, `'UpperSideband'`), construct a minimal `BCFid`
  (or monkey-patch `fidparams`) and call `apply_lo`; for the
  multi-FID `BCFTMW.process_sideband`, exercise each `which` value
  (`upper`, `lower`, `both`) and each `avg` value (`harmonic`,
  `geometric`) once.
- `test_processing_overrides.py` — for `BCFid.ft`, exercise the
  full kwarg surface (`start_us`, `end_us`, `winf`, `zpf`, `rdc`,
  `expf_us`, `autoscale_MHz`, `units_power`, `frame`) one kwarg at a
  time, asserting no exception and finite output.
- `test_window_dispatch.py` — pure-unit test of the
  `_WINDOW_MAP` / `_WINDOW_INT_MAP` lookups (no fixture required).
- `test_enum_helpers.py` — pure-unit test of the
  `_resolve_enum` helper introduced in change #4 (covers the
  int / int-string / name / unknown-input paths).
- `test_differential_fid.py` — exercises the API added in §9:
  - `get_differential_fid()` with defaults returns a `BCFid` whose
    `fidparams.shots` equals `fidparams.iloc[0].shots`.
  - `get_differential_fid(start=k, end=-1).fidparams.shots`
    equals `fidparams.iloc[0].shots − fidparams.iloc[k].shots`
    for each backup index `k` in the fixture.
  - `get_differential_fid(start=i, end=j).fidparams.shots`
    equals `fidparams.iloc[j].shots − fidparams.iloc[i].shots`
    for at least one ordered pair with `0 < i < j`.
  - `get_differential_fid(0, 1).fidparams.shots`
    equals `fidparams.iloc[1].shots`.
  - The returned `BCFid.ft()` runs without raising.
  - Out-of-range `start` / `end`, `start >= end` after resolving
    `end=-1`, and a stub experiment with `FtmwType=LO_Scan` each
    raise `ValueError`.
  - Runs against both `mtbe/` and `v2-ftmw/` fixtures.
- `test_error_paths.py` — error-handling sweep coverage:
  - `header_value` with an absent `(ObjKey, ValueKey)` raises
    `KeyError`.
  - `header_unit` with an absent `(ObjKey, ValueKey)` raises
    `KeyError`; with a row whose unit cell is empty returns `""`
    without raising.
  - `BCFid.ft(winf="NoSuchWindow")` raises `ValueError`; the same
    holds for unknown `FtUnits` (parametrize a fid whose
    `processing.csv` has been monkey-patched to an unknown name).
  - `BCFTMW.get_fid(num=999)` raises `ValueError`.
  - A constructor invoked on a directory missing
    `header.csv`'s `Experiment / Number` row raises
    `KeyError` (or `ValueError`, whichever the implementation
    chose; document and assert the same).

Tests use `pytest`, no other test deps. Add `pytest` to a
`dev-requirements.txt` (or `[project.optional-dependencies]` table in
`pyproject.toml`); do NOT add it to runtime `dependencies`.

Bundle 04 will add LIF-side tests in the same style; this bundle's
tests cover only the FTMW + experiment-top-level surface.

## Files touched

- `python/blackchirp/src/blackchirp/blackchirpexperiment.py`
- `python/blackchirp/src/blackchirp/bcftmw.py` (includes the
  differential-FID API and `is_multi_segment` / `num_backups`
  helpers from §9)
- `python/blackchirp/src/blackchirp/bcfid.py`
- `python/blackchirp/src/blackchirp/_enum_helpers.py` (new)
- `python/blackchirp/tests/` (new directory tree)
- `python/blackchirp/pyproject.toml` — add `[project.optional-dependencies]`
  for `dev` (pytest); no runtime dep changes.

## Acceptance criteria

- `mtbe/` fixture: `BCExperiment(...)` loads, every public method on
  `BCExperiment` / `BCFTMW` / `BCFid` returns without raising on
  default arguments.
- `v2-ftmw/` fixture: `BCExperiment(...)` loads, `markers` is
  populated when present, `BCFTMW.process_sideband(which='lower')`
  produces the expected branch behavior, `BCFid.ft()` resolves the
  string-form `BlackmanHarris` window without falling through to
  boxcar.
- `hardware.csv` reader accepts both `subKey` and `driver` second-
  column headers, and tolerates the `hardwareType` column being
  numeric, string, or absent.
- All seven window functions and all four `FtUnits` values exercise
  through `BCFid.ft()` without raising and produce finite output.
- All four sideband forms (`0`, `1`, `'LowerSideband'`,
  `'UpperSideband'`) route through `apply_lo` and
  `BCFTMW.process_sideband` correctly.
- `BCFTMW.get_differential_fid` returns FIDs whose `shots`
  attribute matches the arithmetic on `fidparams.csv` for both
  fixtures, raises `ValueError` on multi-segment experiments and
  on out-of-range / non-monotone bounds, and produces output that
  runs through `.ft()` without error.
- The error-handling sweep is complete: `header_value` /
  `header_unit` raise `KeyError` on missing rows;
  `BCFid.ft` raises `ValueError` on unknown window or `FtUnits`
  values; every public method's docstring `Raises:` block
  matches the methods's actual behavior.
- `pytest python/blackchirp/tests/` passes.
- `pylint -E python/blackchirp/src/blackchirp/` clean.
- `black --check python/blackchirp/src/blackchirp/` clean.

## Out of scope

- Any LIF read or processing — bundle 04.
- Documentation pages and notebooks — bundle 05.
- Reorganizing the public API surface (e.g. renaming
  `process_sideband`). The bundle preserves existing public
  signatures.
