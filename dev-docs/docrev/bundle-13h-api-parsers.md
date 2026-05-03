# Bundle 13h — API Reference: File Parsers

**Status:** complete

<!--
Status log:
- 2026-05-02: in progress → complete. Content commit 221344ec.
  Also normalized `GenericXYParser::fileExtensions()` to the
  glob-prefixed form matching SPCAT/XIAM so that
  `FileParserRegistry::fileDialogFilter()` produces a valid
  filter string across all registered parsers.
- 2026-05-02: not started → in progress. Scope expanded to add
  `catalogparser.rst` (intermediate abstract base between
  `FileParser` and `SPCATParser`/`XIAMParser`; documents the
  `parse() → CatalogData` hook). `FileParserRegistry` confirmed
  to exist; `fileparserregistry.rst` is in scope.
-->

Adds API reference pages for the catalog and generic-XY file
parsers used by the overlay feature and by experiment loading.

## Scope

New pages under `doc/source/classes/`:

- `fileparser.rst` ← `src/data/processing/parsers/fileparser.h`
  (`FileParser` base class).
- `catalogparser.rst` ←
  `src/data/processing/parsers/catalogparser.h`
  (intermediate abstract base for spectroscopic catalog
  parsers; adds the `parse() → CatalogData` hook on top of
  `FileParser`).
- `spcatparser.rst` ←
  `src/data/processing/parsers/spcatparser.h`.
- `xiamparser.rst` ←
  `src/data/processing/parsers/xiamparser.h`.
- `genericxyparser.rst` ←
  `src/data/processing/parsers/genericxyparser.h`.
- `fileparserregistry.rst` ←
  `src/data/processing/parsers/fileparserregistry.h`
  (singleton registry; `CatalogParserRegistry` was renamed to
  `FileParserRegistry` in commit `124bdaa8`).

`GenericXYData` (the value type returned by
`GenericXYParser::parse()`) does **not** get its own page; the
`genericxyparser.rst` prose names it via inline
`:cpp:class:`GenericXYData`` and the Doxygen output on that page
covers the rest.

## Out of scope

- Overlay UI surfaces.

## Sources

- The header files.
- `doc/source/user_guide/overlays.rst` — confirm the user-facing
  format names match the parser class names (SPCAT vs. SPFIT
  etc.).

## Sphinx file deltas

**Created:** one per page above.

**Possibly modified (Doxygen comment refresh):**
- All headers listed above.

## Acceptance criteria

- `FileParser` base class page documents the hook points a new
  parser must implement.
- Each concrete parser page documents the input file format it
  recognises and the output data shape (so a contributor adding
  a new catalog format can model it on the existing parsers).
- All pages cross-link to the user-guide overlays chapter
  (bundle 09).
