# Bundle 01 — Installation

**Status:** not started

<!--
Status log:
- (entries appended in reverse chronological order; most recent first)
-->

Replace the qmake-era installation page with binary-package coverage
first and a CMake source-build section second.

## Scope

- Rewrite `doc/source/user_guide/installation.rst` end to end. All
  references to `config.pri`, qmake, `qmake6 blackchirp.pro`, and the
  Qt Creator-only build flow are removed.
- Document the binary-distribution path: where to download
  per-platform packages from GitHub Releases (DEB, RPM, AppImage, DMG,
  NSIS), how to install each, and the runtime expectations (Qt, GSL,
  Qwt are bundled or auto-resolved).
- Document the CMake source build: prerequisites, `cmake . -B build`,
  build types, `cmake/BuildConfig.cmake` for optional modules, the
  `breathe` env for docs, where to find binaries afterwards. The
  source-build section is for users who need to enable a hardware
  implementation that the binary distribution does not include.
- Add a short "Optional modules and build options" subsection covering
  `BC_BUILD_DOCUMENTATION`, `BC_ENABLE_LIF`, `BC_ENABLE_CUDA`, plus a
  note on hardware enable flags and that LIF is now a runtime toggle
  in addition to a build option.
- Replace the "Hardware Implementations" subsection with a brief
  pointer to the runtime hardware configuration system (forward link
  to bundle 03's pages); installation no longer selects hardware
  implementations.

## Out of scope

- Per-device installation notes (vendor library paths, udev rules,
  GPIB controller setup) — those live on the per-device pages
  (bundle 05) and the Communication page (bundle 04).
- Detailed runtime hardware configuration — bundle 03.

## Sources

- `dev-docs/packaging-and-ci.md` — package matrix, what each artifact
  contains, supported distros.
- `dev-docs/labjack-cross-platform-support.md` — note on Windows
  vendor-DLL expectations (forward to bundle 04 for the library
  status widget detail).
- `cmake/BuildConfig.cmake` — read for the canonical option list.
- `CLAUDE.md` (project root) — read for the canonical CMake build
  invocations.

## Sphinx file deltas

**Modified:**
- `doc/source/user_guide/installation.rst` — full rewrite.

## Toctree delta

None at the chapter level. The page already exists in
`user_guide.rst`'s toctree.

## Screenshots

- `_static/user_guide/installation/release-page.png` — GitHub Releases
  page showing per-OS artifacts (decorative).
- Optional: `_static/user_guide/installation/cmake-buildconfig.png` —
  excerpt of `BuildConfig.cmake` highlighting the user-tunable
  options.

## Acceptance criteria

- The page contains no occurrence of "qmake", "config.pri",
  "blackchirp.pro", or "Qt Creator IDE" (the latter may be mentioned
  briefly as a CMake-friendly editor option but not as the only path).
- A new user reading the page can complete a binary install on any
  one of the four supported platforms without consulting source.
- The CMake source-build section describes both Debug and Release
  configurations using the canonical command from `CLAUDE.md`.
- All cross-references to other user-guide pages use `:doc:` (not
  raw HTML).
