# Documentation tree — Agent Guide

Sphinx + Doxygen documentation for Blackchirp, deployed to
https://blackchirp.readthedocs.io/. This file applies to all work under
`doc/source/`. Cross-cutting rules (timeless commits, no-install-without-
consent, etc.) are in the repository-root `AGENTS.md`.

## Build

```bash
cmake --build build --target docs     # Sphinx HTML + Doxygen
cmake --build build --target doxygen  # Doxygen XML/HTML only
```

The `docs` target requires Sphinx, Breathe, Doxygen, sphinx_rtd_theme,
nbsphinx, and nbsphinx-link. The pip install set is in
`doc/source/requirements.txt`. Activate an environment that satisfies
these requirements before invoking the build; per-machine activation
details (specific env name, conda vs. pip, etc.) live in
`.claude/CLAUDE.local.md` and are not committed.

If the activation convention is not recorded locally, ask the user
before installing dependencies into the active interpreter or creating
a new environment.

After adding or removing a page, force a toctree rebuild with:

```bash
touch doc/source/index.rst
```

before re-running the docs target. Without this, Sphinx may skip
regeneration and the new page will not link from the chapter index.

Output: `build/docs/html/` (HTML), `build/docs/doxygen/` (API).

## Style rules

### Voice and tense

User-facing prose is present-tense and impersonal ("Blackchirp writes
the FID to disk", not "Blackchirp will write" or "we write"). The
developer guide is the same except second person is acceptable when
giving a contributor explicit instructions ("Add the driver header to
the aggregator").

### Temporal markers

Do not use source-evolution temporal markers in prose: no "Phase 2",
"v1.1.0 introduced", "recently added", "now uses", "previously did X
but now does Y". Permanent version-keyed information lives in the
changelog or migration guide, not in the user or developer guide.
Rendering describing **runtime behavior** is fine ("after the
experiment completes", "before any FID is acquired").

### American English

All prose uses American English: `normalize`/`normalization`,
`behavior`, `color`, `visualization`, `analyze`, `co-averaging`. Match
UI labels exactly when quoting them, even if the UI label uses an
American spelling — do not "correct" a label such as "Randomize Delay
Order" to British spelling in prose.

### Cross-references

Use Sphinx `:doc:` and `:ref:` directives, not raw HTML anchors.
Replace any `<page.html>`-style links opportunistically when editing a
page.

### Index entries

Every new page begins with a `.. index::` block listing the user-facing
terms it introduces.

### Screenshots

UI screenshots go directly in `doc/source/_static/user_guide/` as a
flat directory. Filenames follow `<page>-<topic>.<ext>`, where
`<page>` matches the basename of the `.rst` page that uses the
screenshot (use `hw-<device>` for per-device pages under
`user_guide/hw/`). The hyphen separates page prefix from topic; the
topic is the descriptive part. Reference screenshots with `.. figure::`
directives so they get a caption and fit into the page flow. Sizing
convention: native ≤800 px renders 1:1 (no `:width:`); native >800 px
caps at `:width: 800` with `:target:` linking to the full-resolution
image so users can click through. Match the existing rendered pages.

### API reference

Prefer `.. doxygenclass::` over `.. doxygenfile::` so each class gets a
focused page and member documentation is grouped by member. The
canonical procedure for editing an API page is
`doc/source/developer_guide/api_style.rst`.

### Settings registry

Per-device settings are self-documenting in the UI via the registry's
labels and tooltips. Documentation does not enumerate every setting;
document the non-obvious ones, defaults that matter, and behavioral
caveats. The full registry mechanism is documented on the developer
guide's hardware-configuration page.

## Notebooks

The example notebooks under `python/single-*.ipynb` are referenced from
the docs via `nbsphinx-link`. They must be **fully executed** (with
output cells) before commit, or they will not render correctly.
Execute the notebook in an environment with the published dependency
set; do not commit a notebook with stripped or partial outputs.

## What to read before starting

For any non-trivial doc edit, the canonical references on the docs'
own conventions are:

- `doc/source/developer_guide/api_style.rst` — API page structure,
  Doxygen prose vs. `.rst` page balance, the per-class refresh
  checklist.
- The existing pages in the section you are editing — match their
  voice, depth, and structure rather than introducing a new style.
