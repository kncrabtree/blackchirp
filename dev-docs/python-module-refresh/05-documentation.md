# Bundle 05 — Documentation Refresh, Notebooks, Version Bump

**Status:** not started
**Depends on:** 03, 04
**Effort:** M (2 sessions)

## Scope

Bring the rendered Sphinx documentation for the Python module into
line with the per-class layout used everywhere else in the 2.0 docs,
add a LIF example notebook, and ship a version bump.

## Layout target

The C++ API reference settled on a per-class layout during the 2.0
documentation rebuild — see `doc/source/classes/blackchirpcsv.rst`
for the canonical shape:

1. `index::` directive block listing the relevant index entries.
2. Class name as the page title.
3. A short prose intro (1–4 paragraphs) covering the class's role,
   how it relates to neighbors, and any non-obvious invariants. Cross-
   link to user-guide pages where appropriate.
4. An `API Reference` section ending in the autodoc directive
   (`doxygenclass` for C++, `autoclass` for Python).

The Python module's current page (`doc/source/python.rst` +
`python/docs-index.rst`) lumps every class onto one page and is
missing `BCLIF` from its autoclass list entirely. This bundle splits
the page into per-class files matching the C++ convention.

## Files to create

Under `doc/source/python/`:

- `bcexperiment.rst` — overview of the experiment loader (path
  conventions, what each attribute represents, how it relates to the
  user-guide data-storage page). Cross-link to
  `/user_guide/data_storage`.
- `bcftmw.rst` — CP-FTMW data container, multi-FID processing,
  sideband deconvolution. Cross-link to
  `/user_guide/cp-ftmw#sideband-deconvolution`.
- `bcfid.rst` — single FID + frame layout, FT processing settings,
  override-vs-default behavior of `ft()`.
- `bclif.rst` — `BCLIF` scan container. Cross-link to
  `/user_guide/lif/data_storage`.
- `bcliftrace.rst` — `BCLifTrace` per-point trace + smoothing +
  integration semantics.

Each page follows the layout target above and ends with
`.. autoclass:: blackchirp.<ClassName>` with `:members:`.

## Files to update

- `doc/source/python.rst` — replace the body with a glob-style toctree
  pointing at the per-class pages, mirroring `doc/source/classes.rst`.
  Keep a short prose intro covering `pip install blackchirp`, the
  `from blackchirp import *` recommendation, and a one-line pointer to
  the example notebooks.
- `doc/source/python/docs-index.rst` — delete (replaced by the
  per-class pages and the new toctree on `python.rst`).
- `doc/source/python/example.rst` — keep, but expand the toctree
  caption to cover both notebooks; rename to `examples.rst` if a
  prose intro is added.
- `python/blackchirp/src/blackchirp/__init__.py` module docstring —
  add `BCLifTrace` to the class list, add a one-paragraph LIF quick-
  start mirroring the FTMW example.
- `python/blackchirp/pyproject.toml` — bump `version` to `0.1.0` (or
  whatever the maintainer chooses; the schema and API surface have
  changed enough that the bump is warranted).
- `python/blackchirp/README.md` — refresh if it references v1 schema.

## Notebooks

- `python/single-fid.ipynb` — re-execute against
  `python/example-data/v2-ftmw/`. Update inline prose if it references
  v1 schema specifics (e.g. numeric `sideband == 1`). Keep narrative
  beats: load → inspect header → compute FT → tweak processing →
  re-FT.
- `python/single-lif.ipynb` — new notebook, fully executed against
  `python/example-data/v2-lif/`. Cover: load experiment, list
  available scan points, fetch one `BCLifTrace`, smooth + integrate
  with default and override gates, compute a `delay_slice`, compute
  the full `image()`, and walk through the `fill=` option for
  missing points. Use matplotlib for visualization throughout
  (single-trace plot, slice plot, and a 2-D pcolormesh / imshow of
  the image grid). Matplotlib is a notebook-only dependency and is
  *not* added to the runtime `pyproject.toml` requirements.
- `doc/source/python/notebooks/single-fid.nblink` — verify path still
  resolves after any path rearrangement; no change expected.
- `doc/source/python/notebooks/single-lif.nblink` — new, points at
  `python/single-lif.ipynb`.

CONTRIBUTING.md requires that example notebooks be fully executed in
their committed form, so re-run both notebooks end-to-end before
committing.

## Style requirements

Borrowed from the 2.0 documentation rebuild:

- **Timeless prose.** No "previously", "now", "as of v0.x" with
  respect to source evolution. Runtime/state language is fine.
- **Cross-link to the user guide.** Every class page should point at
  the relevant user-guide section so readers do not have to search
  for the on-disk schema description.
- **One module-level intro paragraph per page.** The autoclass
  directive provides the per-method detail; the page intro covers the
  *why*.
- **No emojis.**

## Files touched

- `doc/source/python.rst` (rewritten)
- `doc/source/python/bcexperiment.rst`,
  `doc/source/python/bcftmw.rst`,
  `doc/source/python/bcfid.rst`,
  `doc/source/python/bclif.rst`,
  `doc/source/python/bcliftrace.rst` (new)
- `doc/source/python/docs-index.rst` (deleted)
- `doc/source/python/example.rst` (kept or renamed)
- `doc/source/python/notebooks/single-lif.nblink` (new)
- `python/blackchirp/src/blackchirp/__init__.py` (docstring + export)
- `python/blackchirp/pyproject.toml` (version bump)
- `python/blackchirp/README.md` (refresh as needed)
- `python/single-fid.ipynb` (re-executed against v2 fixture)
- `python/single-lif.ipynb` (new, fully executed)

## Acceptance criteria

- `cmake --build build --target docs` (per the standard doc-build
  recipe in CLAUDE.md) succeeds with no new Sphinx warnings.
- Rendered HTML shows one page per Python class, each with an
  `index::`-driven entry in the genindex.
- Both notebooks render through `nbsphinx` with their committed cell
  outputs (no "this cell has not been executed" placeholders).
- Module docstring (`__init__.py`) lists all five public classes
  including `BCLIF` and `BCLifTrace`.
- `pyproject.toml` version is bumped and the README does not contain
  v1-only language.

## Out of scope

- A standalone tutorial or how-to chapter beyond the two notebooks.
- Coverage of `BCExperiment` plotting (the module does not plot).
- Any further restructure of the C++ API pages.
