# Bundle 02 — Refresh `python/example-data/` Fixtures

**Status:** not started
**Depends on:** —
**Blocks:** 03, 04, 05
**Effort:** S (1 session)

## Scope

Provide fixtures that the Python module's notebooks, tests, and
docstrings can run against. Today the only fixture is
`python/example-data/mtbe/`, which is a Blackchirp 1.0 experiment and
does not exercise any v2 schema changes (label-based hardware keys,
`markers.csv`, string-form enums) or LIF data at all.

This bundle adds two new fixtures alongside `mtbe/`, so that the
backward-compat smoke-test value of the old fixture is preserved while
the v2 schema is also covered.

## What to add

Pick from the recent runs in `data/experiments/0/0/` (current dev-box
data, not yet shipped to any user). Suggested candidates:

- **FTMW v2 fixture**: `data/experiments/0/0/53/` — single-FID Forever
  acquisition, label-based hardware keys, current `processing.csv`
  with `BlackmanHarris` window function. Copy as
  `python/example-data/v2-ftmw/`.
- **LIF v2 fixture**: `data/experiments/0/0/52/` (or `51/`) — partial
  6×6 grid with reference channel, exercises the missing-scan-point
  case that `BCLIF.image()` will need to handle.
  Copy as `python/example-data/v2-lif/`.

If a current run with `markers.csv` is available, prefer it over the
candidates above. If none exists, capture one from a virtual AWG that
supports markers and add it; failing that, document in the bundle that
`markers.csv` coverage will need to be added when a marker-capable
acquisition is run.

Retain `python/example-data/mtbe/` unchanged as the v1 backward-compat
smoke-test fixture.

## Files touched

- `python/example-data/v2-ftmw/` — new directory tree (full copy of
  the source experiment).
- `python/example-data/v2-lif/` — new directory tree.
- `python/example-data/README.md` — new short index file describing
  what each fixture covers, the Blackchirp version it was written
  with, and any notable schema features it exercises (e.g. "v2-lif
  has reference channel enabled, partial scan").

## Acceptance criteria

- The two new fixtures load via the current (un-refreshed) Python
  module without raising on the FTMW data path. (The LIF data path is
  expected to silently load only `lifparams.csv` / `processing.csv`
  until bundle 04 lands.)
- The `mtbe/` fixture continues to load unchanged.
- `README.md` accurately describes each fixture's schema features so
  downstream bundles can target their tests.

## Notes

- This bundle should land *after* bundle 01 has flipped the C++
  writers, so that the v2 fixtures captured here use the final
  string-form representation for `FtUnits` and `hardwareType`. If
  bundle 02 is captured before bundle 01 lands, re-record the v2
  fixtures (or hand-edit the affected cells) so that the on-disk
  format matches what the module will encounter going forward.
- Do not include large multi-FID datasets. Single-FID FTMW and a
  small LIF grid are sufficient for the notebook + test surface.
