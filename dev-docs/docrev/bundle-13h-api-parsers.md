# Bundle 13h — API Reference: File Parsers

Adds API reference pages for the catalog and generic-XY file
parsers used by the overlay feature and by experiment loading.

## Scope

New pages under `doc/source/classes/`:

- `fileparser.rst` ← `src/data/processing/parsers/fileparser.h`
  (`FileParser` base class).
- `spcatparser.rst` ←
  `src/data/processing/parsers/spcatparser.h`.
- `xiamparser.rst` ←
  `src/data/processing/parsers/xiamparser.h`.
- `genericxyparser.rst` ←
  `src/data/processing/parsers/genericxyparser.h`.

If a `FileParserRegistry` exists (commit `124bdaa8` mentions a
parser registry rename), add a `fileparserregistry.rst` page as
well.

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
