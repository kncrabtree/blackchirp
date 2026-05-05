# Python Module Refresh — Master Plan

The companion Python module under `python/blackchirp/` was last updated on
the `devel` branch and predates a number of file-format changes that
landed during the 2.0 development cycle: label-based hardware keys, the
new `markers.csv`, schema additions to `hardware.csv`, the migration of
several enum values to string-name representations, and the promotion of
the LIF module to a first-class Blackchirp feature. The module also has
no LIF read or processing support at all.

This plan groups the refresh into self-contained **bundles** under
`dev-docs/python-module-refresh/`. Each bundle file states its scope,
inputs, files it touches, and acceptance criteria. The scope of this
project is much smaller than the documentation overhaul, so this plan
intentionally omits the orchestrator/drafter-verifier scaffolding from
`dev-docs/documentation-revision.md`; the bundles can be picked up by a
single contributor in sequence.

## Project layout

- `dev-docs/python-module-refresh/00-overview.md` — this file (master plan)
- `dev-docs/python-module-refresh/NN-name.md` — one file per work bundle
- `python/blackchirp/` — Python module source (target of bundles 03–05)
- `python/example-data/` — fixtures for module tests and notebooks
- `src/data/storage/`, `src/data/analysis/`, `src/hardware/` — C++ writers/readers touched by bundle 01
- `doc/source/python/` — Sphinx pages (target of bundle 05)

## Bundles at a glance

| ID | Title | Effort | Depends on | Status |
|----|-------|--------|------------|--------|
| 01 | C++ enum-string migration and reader hardening | M | — | in progress |
| 02 | Refresh `python/example-data/` fixtures | S | — | not started |
| 03 | Python module schema fixes (FTMW + experiment top level) | M | 01, 02 | not started |
| 04 | `BCLIF` and `BCLifTrace` implementation | L | 01, 02 | not started |
| 05 | Documentation refresh, notebook updates, version bump | M | 03, 04 | not started |

Effort key: S ≈ 1 session, M ≈ 2 sessions, L ≈ 3+ sessions.

Status values:

- **not started** — no work has been done on this bundle.
- **in progress** — the bundle has been opened but is not yet
  complete. The bundle's own header carries the detail (open
  questions, partial-landing summary).
- **complete** — the bundle's content commit has landed. The commit
  hash is recorded in the bundle's own status header for
  traceability.
- **blocked** — work has been attempted but cannot proceed without a
  human decision. The blocker is described in the bundle's header.

This table is the **single source of truth for progress.** Update it
(via `Edit`) when a bundle's status transitions.

## Cross-cutting style requirements

Borrowed from `dev-docs/documentation-revision.md` and the project
CLAUDE.md, applied to every bundle:

- **Timeless prose.** Comments and commit messages must read correctly
  to a developer five years from now with no knowledge of the commit
  in front of them. No "Phase X", "now", "previously", "added in
  vN.M" with respect to source evolution. Runtime/state language
  ("the first FID arrives", "previously stored configuration") is
  fine.
- **No bare numeric enum literals in new code.** When introducing or
  modifying a writer for a `Q_ENUM`-bearing field, wrap with
  `QVariant::fromValue(e)` and let `QVariant::toString()` produce the
  enum name. When introducing or modifying a reader, accept either
  the integer or the name (see bundle 01 for the canonical helpers).
- **Python style.** `black` formatting and `pylint -E`-clean. Google-
  style docstrings on every public class and function. Example
  notebooks fully executed in their committed form (nbsphinx renders
  the cached outputs).
- **Minimal Python dependencies in the module itself.** numpy, scipy,
  and pandas only. No matplotlib, no plotting, no peak finding inside
  the `blackchirp` package — out of scope per long-standing project
  convention.
- **Notebook visualization is encouraged.** The example notebooks
  (`single-fid.ipynb`, `single-lif.ipynb`) may import matplotlib and
  use it for plotting, mirroring the style of the existing FTMW
  notebook. Notebook-only dependencies are documented in the notebook
  itself, not added to `pyproject.toml` runtime requirements.

## Conda environments

Two separate conda environments back this work; do not mix them.

- **`blackchirp-py`** — Python module development, testing, linting,
  formatting, notebook execution, and PyPI build/publish. Defined by
  `python/environment.yml` (mirror in `python/requirements.txt` for
  pure-pip workflows). Run any Python-side command through it:

  ```bash
  conda run -n blackchirp-py pytest python/blackchirp/tests/
  conda run -n blackchirp-py pylint -E python/blackchirp/src/blackchirp/
  conda run -n blackchirp-py black --check python/blackchirp/src/blackchirp/
  conda run -n blackchirp-py jupyter nbconvert --to notebook --execute python/single-fid.ipynb
  conda run -n blackchirp-py python -m build python/blackchirp
  ```

  The environment already has the module installed in editable mode
  (`pip install -e python/blackchirp`), so source edits are visible to
  `pytest` and notebook runs without reinstalling. Re-run the editable
  install only after the environment is recreated, after the package
  layout changes, or after `pyproject.toml` metadata
  (build-backend, `[project]`, or `dependencies`) changes.

- **`breathe`** — Sphinx + Doxygen documentation builds only. Doc
  rebuilds follow the recipe from `dev-docs/documentation-revision.md`
  and the project CLAUDE.md:

  ```bash
  touch doc/source/index.rst && conda run -n breathe cmake --build build --target docs
  ```

  Use this for bundle 05's doc rebuild and any spot-check of rendered
  output.

C++ work in bundle 01 does not need a conda environment — build via
`cmake --build build/Desktop-Debug -j$(nproc)` and test via
`ctest --test-dir build/tests` per the standard recipes in CLAUDE.md.

## Cross-cutting safety rule

Any C++ reader of an enum-bearing CSV cell must accept both numeric
and name representations. This is a defensive requirement: the on-disk
representation has changed across versions, and Blackchirp must read
its own historical output. Bundle 01 audits and hardens the reader
side first; bundles 03 and 04 then assume the dual-form invariant
holds.

## Open questions

None at this time. Decisions captured during planning:

- `FtUnits` migrates from numeric to string in `processing.csv`
  (`FtUnits;FtuV` instead of `FtUnits;6`). C++ reader accepts both;
  Python module maps name → exponent.
- `hardware.csv`'s `hardwareType` column is under active review. It
  may end up as a string-form enum, or it may be removed entirely as
  redundant (it encodes the numeric value of the driver-type enum
  for programmatic convenience only — the `subKey` column already
  carries the user-facing driver identity). Bundle 01 makes the call
  during the audit; the defensive reader is built either way.
- LIF API exposes `fill=np.nan` (default) or `fill=0` on aggregating
  helpers (`image()`, `delay_slice()`, `laser_slice()`) to handle
  missing scan points.
- Module gains a fully-executed `single-lif.ipynb` example notebook
  alongside the refreshed `single-fid.ipynb`.
