# Bundle 12a — Developer Guide: Build System & Project Layout

**Status:** not started

<!--
Status log:
- (entries appended in reverse chronological order; most recent first)
-->

First sub-page of the Developer Guide chapter (bundle 12). Explains the
CMake-driven build system: the layout of `cmake/*.cmake` modules, the
`BuildConfig.cmake` user-options file, the auto-generated `hw_*.h`
hardware aggregator headers, optional modules (CUDA, Python hardware,
LIF, docs, tests), the viewer-only build, and developer-level
packaging via CPack.

This page is foundational: it has no developer-guide dependencies and
should be readable in isolation. The user-facing source-build steps
already live in `doc/source/user_guide/installation.rst`; the
developer-guide page is one level deeper — it explains *why* the CMake
layout is shaped the way it is and how a contributor adds to it.

## Scope

Single Sphinx file: `doc/source/developer_guide/build_system.rst`.

The page should answer the following questions for a contributor:

1. **What does each `cmake/*.cmake` module do, and which library/target
   does it produce?** Walk the module list:

   - `BlackchirpData.cmake` → `blackchirp-data` static library (data
     model, storage, analysis, settings, logging — used by both the
     main app and the viewer).
   - `BlackchirpHardware.cmake` → `blackchirp-hardware` static library
     (hardware base classes, all implementations, communication
     protocols, registration, vendor libraries, Python trampolines).
     Only built when `BC_BUILD_VIEWER_ONLY=OFF`.
   - `BlackchirpGui.cmake` → `blackchirp-gui` static library (full GUI
     widgets, plots, dialogs). Main app only.
   - `BlackchirpViewerGui.cmake` → `blackchirp-viewer-gui` static
     library (lighter GUI subset for the viewer; no hardware
     dependencies).
   - `BlackchirpApplication.cmake` → `blackchirp` executable, glues
     data + hardware + gui together, applies `blackchirp_deploy_qt()`.
   - `BlackchirpViewerApplication.cmake` → `blackchirp-viewer`
     executable, uses data + viewer-gui only.
   - `BlackchirpDocumentation.cmake` → Doxygen + Sphinx targets.
   - `Packaging.cmake` → CPack configuration (DEB/RPM/TGZ on Linux,
     DragNDrop/TGZ on macOS, NSIS/ZIP on Windows; component
     restriction to `Applications` only).
   - `QtDeployment.cmake` → `blackchirp_deploy_qt(<target>)` install
     hook that wraps `windeployqt`/`macdeployqt` (no-op on Linux).
   - `FindQWT.cmake` → custom finder for the Qwt scientific plotting
     library (no Qt6 Config file ships with most distros).

2. **What's in `BuildConfig.cmake` and how does it work?** Cover:

   - The auto-creation pattern from `BuildConfig.cmake.template` on
     first CMake run (the file is git-ignored so user changes persist
     across pulls).
   - The user-facing options:
     `BC_ENABLE_CUDA`,
     `BC_BUILD_VIEWER_ONLY`,
     `BC_BUILD_TESTS`,
     `BC_BUILD_DOCUMENTATION`.
   - That hardware *selection* is runtime via the registry/profile
     system, not compile-time. The build always compiles every
     hardware implementation; the active set is decided at run time.

3. **The auto-generated hardware aggregator headers.** Explain:

   - `BlackchirpHardware.cmake` writes three headers at configure time
     into `src/hardware/core/`:
     `hw_base.h` (interface classes — `clock.h`, `ftmwscope.h`, `awg.h`,
     `pulsegenerator.h`, …),
     `hw_impl.h` (every concrete implementation header, including
     Python trampoline headers),
     `hw_h.h` (combines both).
   - These exist so AUTOMOC pulls the meta-object code for every
     implementation and the static `REGISTER_HARDWARE_*` initializers
     are not silently dropped from the static library at link time.
   - Glob-based discovery: dropping a new `xxxx.cpp`/`xxxx.h` pair into
     `src/hardware/core/<type>/` or `src/hardware/optional/<type>/`
     under the recognized name patterns is enough — no manual
     CMakeLists.txt edit is needed. (List the patterns: `virtual*`,
     `dsa*`, `m4i*`, `dso*`, `dpo*`, `mso*`, `valon*`, `hp*`, `awg*`,
     `ad*`, `m8*`, `qc*`, `bnc*`, `srs*`, `mks*`, `prologix*`,
     `labjack*`, `intellisys*`, `lakeshore*`, `rigol*`, `opolette`,
     `sirah*`, `fixedclock`.)
   - Python hardware: `BlackchirpHardware.cmake` also globs
     `src/hardware/python/*.cpp` (trampolines) and
     `src/hardware/python/python_*_template.py` + `python_hw_host.py`
     (runtime files). The runtime files are `configure_file`d into the
     build directory and `install`ed under `share/blackchirp/`.

4. **Build directories and recommended commands.** Mirror the
   project's CLAUDE.md conventions exactly:

   - `cmake . -B build/Desktop-Debug/` and
     `cmake --build build/Desktop-Debug/ -j$(nproc)`.
   - `cmake --build` (not `make -C`) so CMake regeneration mid-build
     does not break.
   - Targets: `blackchirp`, `blackchirp-viewer`, `tests`, `docs`,
     `doxygen`.
   - The Sphinx build's reliance on the `breathe` conda environment
     and the `touch doc/source/index.rst && conda run -n breathe cmake
     --build build --target docs` workaround for stale Breathe XML.

5. **Test targets.** Walk the contents of the top-level `if(BC_BUILD_TESTS)`
   block in `CMakeLists.txt`:

   - The list of test executables and what each covers (one short line
     per: `tst_settingsstoragetest`, `tst_headerstoragetest`,
     `tst_blackchirpcsvtest`, `tst_ftworkertest`,
     `tst_genericxyparser`, `tst_operation_capability_only`,
     `tst_overlayoperations_simple`, `tst_spcatparser`,
     `tst_xiamparser`, `tst_hardwareregistrytest`,
     `tst_runtimehardwareconfigtest`,
     `tst_hardwareprofilemanagertest`, `tst_hardwarekeys`,
     `tst_experimentloading`, `tst_scientificspinboxtest`,
     `tst_zoompanplotthreadsafety`, `tst_waveformbuffertest`,
     `tst_loadoutmanagertest`).
   - The `blackchirp-test-hardware` static library — why it exists
     (avoids `HardwareManager` symbol conflicts in tests that need
     just the virtual implementations) and what it includes.
   - Adding a new test: `add_executable` + `target_link_libraries`
     against `blackchirp-data` (or `blackchirp-test-hardware` if
     virtual hardware is needed) + `Qt6::Test` + `add_test`. Add the
     new target to the `tests` custom-target dependency list.
   - Running tests: `ctest --test-dir build/tests` or
     `ctest --test-dir build/Desktop-Debug` depending on which build
     directory enabled `BC_BUILD_TESTS`.

6. **Documentation build.** Cover:

   - Activating `BC_BUILD_DOCUMENTATION`.
   - The breathe conda environment requirement; the
     `touch doc/source/index.rst` + `conda run -n breathe …` recipe
     (lifted directly from `MEMORY.md`).
   - Output locations: `build/docs/html/` and `build/docs/doxygen/`.
   - That the Sphinx config and Doxygen config (`Doxyfile.in`) live
     in `doc/source/` and are configured via `BlackchirpDocumentation.cmake`.

7. **Packaging at a developer level.** Self-contained tour:

   - CPack-driven on every platform, plus `linuxdeploy` for AppImage.
   - `CPACK_COMPONENTS_ALL` restricted to `Applications` so static
     libraries and dev headers do not ship in binary packages.
   - `blackchirp_deploy_qt()` install hook for `windeployqt` /
     `macdeployqt`.
   - The five `.github/workflows/release.yml` jobs (linux-deb,
     linux-rpm, linux-appimage, macos-dmg, windows-nsis) and how to
     dispatch them.

## Out of scope

- User-facing "how to build from source" steps — already covered in
  `doc/source/user_guide/installation.rst`. Cross-link with `:doc:`,
  do not duplicate.
- Hardware-driver authoring — bundle 12l covers that as a separate
  page; only mention the build-system aspects (glob pattern, no
  manual CMake edit) here and forward-link with `:doc:`.
- The full BuildConfig option enumeration with prose for each — the
  template file is the canonical reference. The page should describe
  the important options inline.

## Sources

Read bundle 12 (the chapter umbrella) before drafting; it carries the
cross-cutting conventions (research-vs-link policy, dev-docs
treatment, gap-flagging protocol, voice/spelling rules) that apply to
every developer-guide sub-page. The lists below are this page's
specific reading list.

### Related source files

- `CMakeLists.txt` (project root) — top-level project, options,
  dependencies, test executable definitions, the
  `blackchirp-test-hardware` library.
- `cmake/BuildConfig.cmake.template` — option declarations and inline
  documentation; canonical reference for the `BC_*` options.
- `cmake/BlackchirpData.cmake`,
  `cmake/BlackchirpHardware.cmake`,
  `cmake/BlackchirpGui.cmake`,
  `cmake/BlackchirpViewerGui.cmake`,
  `cmake/BlackchirpApplication.cmake`,
  `cmake/BlackchirpViewerApplication.cmake`,
  `cmake/BlackchirpDocumentation.cmake`,
  `cmake/Packaging.cmake`,
  `cmake/QtDeployment.cmake`,
  `cmake/FindQWT.cmake` — per-module sources; one of these per
  library/target described in the *CMake module map* section.
- The auto-generated `src/hardware/core/hw_base.h`,
  `src/hardware/core/hw_impl.h`, `src/hardware/core/hw_h.h` —
  inspect to confirm the glob patterns currently in
  `BlackchirpHardware.cmake` produce the documented include set.
- `CLAUDE.md` (project root) — canonical build commands and directory
  conventions; quote where appropriate.

### Related dev-docs

- `dev-docs/packaging-and-ci.md` — full packaging-and-CI reference.
  Read it as research; the *Packaging* section on this page must be
  self-contained and must not link to it.

### Related user-guide pages

- `doc/source/user_guide/installation.rst` — user-facing
  source-build steps; cross-link, do not duplicate.

### Related API reference pages

None directly. The build system is upstream of every API page; the
*Hardware aggregator headers* section may forward-link to
`doc/source/classes/hardwareregistry.rst` since that page documents
the `REGISTER_HARDWARE_*` macros whose static initializers AUTOMOC
linkage protects.

## BC_ALLHARDWARE removal

The `BC_ALLHARDWARE` option is a leftover from the old compile-time
hardware-selection model. Hardware selection is now entirely runtime
via the registry/profile system, and the build always compiles every
implementation regardless of this flag. **Remove it** as part of this
bundle:

1. Delete the `option(BC_ALLHARDWARE …)` declaration from
   `cmake/BuildConfig.cmake.template`.
2. If the same option is declared anywhere else (top-level
   `CMakeLists.txt` or a per-module `.cmake` file), delete those
   declarations too.
3. Delete any `if(BC_ALLHARDWARE)` / `${BC_ALLHARDWARE}` references in
   the cmake tree.
4. Confirm a `grep -rin "BC_ALLHARDWARE"` over the repository returns
   no remaining hits before the bundle is considered complete.
5. Do **not** mention `BC_ALLHARDWARE` in the new RST page; the
   developer guide describes the current build.

This is a small mechanical change but is in scope for the same commit
as the build-system page — it keeps documentation and cmake honest
about each other.

## Sphinx file deltas

**Modified:**

- `doc/source/developer_guide.rst` — add `developer_guide/build_system`
  to the toctree (the placeholder `overview.rst` may be deleted in
  bundle 12 (intro) or in the first sub-page that lands; this page
  should add itself to the toctree without removing siblings).

**Created:**

- `doc/source/developer_guide/build_system.rst`.

## Toctree delta

In `doc/source/developer_guide.rst`, ensure the toctree lists
`developer_guide/build_system` (the chapter intro bundle 12 may have
already done this; if not, add it).

## Page structure

Suggested H2 sections (use `-` underlines, per the API-style
convention):

- *CMake module map* — table mapping each `cmake/*.cmake` module to
  the library/target it produces.
- *Configuring a build* — `cmake . -B …` invocation, build types,
  the `BuildConfig.cmake` auto-creation pattern, the user options.
- *Building targets* — main app, viewer, tests, docs.
- *Hardware aggregator headers* — `hw_base.h` / `hw_impl.h` /
  `hw_h.h`, glob-based source discovery, AUTOMOC linkage requirement.
- *Test infrastructure* — the test list, `blackchirp-test-hardware`,
  recipe for adding a test.
- *Documentation build* — breathe environment, `docs`/`doxygen`
  targets, output locations.
- *Packaging* — self-contained summary covering CPack, the
  `Applications`-only component, the deploy-Qt install hook, and the
  GitHub Actions release jobs.

A 1–2 paragraph H1 intro should set the audience: contributors who
want to extend the build system or understand why the layout is the
way it is, not first-time builders.

## Screenshots

None.

## Acceptance criteria

- Each `cmake/*.cmake` module is named, classified by which target it
  produces, and described in one or two sentences.
- The `BuildConfig.cmake` auto-creation behavior is explained,
  including the git-ignore policy on the generated file.
- The four `BC_*` build options in `BuildConfig.cmake.template`
  (after `BC_ALLHARDWARE` removal) are listed with their effect on
  the build.
- The `hw_base.h`/`hw_impl.h`/`hw_h.h` mechanism is described,
  including the AUTOMOC rationale (without it, static registration
  initializers are silently dropped from the static library).
- Glob source-discovery is explained well enough that a contributor
  knows whether a new file in `hardware/core/<type>/` will be picked
  up automatically (and which name patterns are recognized).
- Build commands match `CLAUDE.md` exactly: full paths or `-C`/`-B`
  flags, `cmake --build` (not `make -C`), `-j$(nproc)`.
- Adding a new test is reduced to a 4-step recipe; the
  `blackchirp-test-hardware` library is mentioned with its purpose.
- The doc build recipe includes the
  `touch doc/source/index.rst && conda run -n breathe cmake --build
  build --target docs` form.
- Packaging coverage is one or two self-contained paragraphs (no
  link into `dev-docs/`).
- The project's evolution from qmake to CMake is fine as a one-line
  aside, but the prose describes the current build, not the history.
- Cross-links to `:doc:`/user_guide/installation`` and the relevant
  API pages where appropriate.

(General conventions — timeless prose, American English, `:doc:`/`:ref:`
cross-references, no rendered links into `dev-docs/` — are inherited
from bundle 12 and are not repeated here.)
