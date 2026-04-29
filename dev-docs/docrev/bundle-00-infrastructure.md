# Bundle 00 — Doc Infrastructure, Landing Page, README

**Status:** drafted

<!--
Status log:
- (entries appended in reverse chronological order; most recent first)
- 2026-04-28: in progress → drafted. All acceptance criteria met:
  docs target builds with zero non-pre-existing warnings; nav order is
  User Guide → Migration → Changelog → Developer Guide → API Reference
  (plus preserved Python Module); README purged of v1.1.0 / qmake /
  config.pri; Sphinx version "2.0.0-alpha" pulled from CMakeLists.txt.
  Also renamed classes.rst H1 from "Developer Guide" to "API Reference"
  to disambiguate the new top-level Developer Guide chapter. Awaiting
  user review and commit.
- 2026-04-28: not started → in progress. Direct (orchestrator)
  handling per master plan; work done in main worktree.
-->

Establishes the Sphinx skeleton, navigation, and project-wide entry
points that every later bundle plugs into.

## Scope

- Refresh `doc/source/conf.py` (project metadata, version pulled from
  CMake, Doxygen XML path under the build tree, breathe extension list).
- Reorganise top-level `doc/source/index.rst` toctree to expose the
  five major chapters that later bundles will populate: User Guide,
  Migration Guide, Changelog, Developer Guide, API Reference.
- Refresh `doc/source/user_guide.rst` to match the post-revision
  toctree (page list comes from later bundles; placeholder entries are
  acceptable in this bundle so the build does not break).
- Add empty changelog scaffold (`doc/source/changelog.rst` plus a
  `doc/source/changelog/` folder) and migration-guide scaffold
  (`doc/source/migration.rst`). Both contain only headings and a brief
  intro; bundle 11 fills them in.
- Add developer-guide scaffold (`doc/source/developer_guide.rst` and
  `doc/source/developer_guide/` folder). Bundle 12 fills it in.
- Refresh `README.md` at the repository root: drop the v1.1.0-preview
  "What's New" entry; replace with a current 2.0.0-alpha summary that
  points to the new docs structure (binary install, runtime hardware
  configuration, Python hardware, Blackchirp-viewer); keep the Discord
  link and Python module section.
- Verify the `cmake --build … --target docs` invocation works on the
  cmakemigration branch with the refreshed `conf.py` (this catches
  Doxygen path mistakes early).

## Out of scope

- Writing the actual user-guide chapter content (later bundles).
- Doxygen XML output paths beyond confirming the existing CMake-driven
  configuration still resolves (already wired via
  `dual Doxyfile generation` work — see `Doxyfile.in`).

## Sources

- `doc/source/conf.py`, `doc/source/index.rst`, `doc/source/user_guide.rst`
- `README.md`
- `CMakeLists.txt` (read-only — pull `BC_MAJOR_VERSION` etc.)
- Existing `cmake/Documentation.cmake` (read-only — confirm Doxygen XML
  output path is what `conf.py` expects)

## Sphinx file deltas

**Modified:**
- `doc/source/conf.py`
- `doc/source/index.rst`
- `doc/source/user_guide.rst`
- `README.md`

**Created:**
- `doc/source/changelog.rst` (scaffold)
- `doc/source/migration.rst` (scaffold)
- `doc/source/developer_guide.rst` (scaffold)
- `doc/source/changelog/` (empty folder, `.gitkeep` if needed)
- `doc/source/developer_guide/` (empty folder)

## Toctree delta in `index.rst`

Before:

```rst
.. toctree::
   :hidden:

   user_guide
   python
   classes
```

After:

```rst
.. toctree::
   :hidden:

   user_guide
   migration
   changelog
   developer_guide
   classes
   python
```

## Screenshots

None.

## Acceptance criteria

- `cmake --build build/Desktop-Debug --target docs` (with the `breathe`
  conda env active) completes without warnings about missing toctree
  entries.
- The HTML build renders five top-level navigation entries (User
  Guide, Migration, Changelog, Developer Guide, API Reference) in the
  order shown above.
- `README.md` no longer mentions v1.1.0 preview, qmake, or
  `config.pri`.
- The version string displayed by Sphinx matches
  `${BC_MAJOR_VERSION}.${BC_MINOR_VERSION}.${BC_PATCH_VERSION}` from
  CMake.
