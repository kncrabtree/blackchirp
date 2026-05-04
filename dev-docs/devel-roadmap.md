# Development Roadmap

Projects sorted by estimated complexity (smallest first). All are largely independent.

## Small

### AddProfileDialog size

When opened, the AddHardwareProfile dialog does not resize to show the contents of the tab widget; the viewport for that widget is only about
100 pixels.

### Lightbox click-through for user-guide screenshots

Replace the static `:target:` click-through wired up by bundle 14b
with a JS lightbox so oversized screenshots open in an in-page modal
overlay rather than navigating away to the raw PNG. The
`sphinxcontrib-lightbox2` extension is the obvious candidate: drop it
into `doc/source/requirements.txt` (so RTD picks it up) and into the
`extensions` list in `doc/source/conf.py`, then convert the
oversized-screenshot directives to its `.. lightbox::` form (or rely
on whatever auto-wrapping the extension does for `.. image::`/
`.. figure::` with a `:target:`). The 30 capped directives that 14b
already shaped are the work surface; the small ones can stay as
plain images. Verify: build clean, RTD preview shows an overlay
rather than a navigation, click-outside / ESC dismisses correctly,
and the keyboard / screen-reader path is still navigable.

- Fix doc warnings

- Update python module for v1/v2 compatability

## Medium

- Add LIF to python module, add example notebook.

## Large

None.

## Pre-Release

### [Documentation Revision](documentation-revision.md)

The sphinx/breathe documentation is outdated and needs to be updated for the
`cmakemigration` branch. The goals are:

- Improve the readme and program summary for the documentation landing page and Github README.
- Update the changelog with major feature developments; establish best practices for keeping it updated
- Prepare a feature summary/migration guide for version 1.x -> 2.0.0.
- Update the user guide to provide a walkthrough of major program features and use-cases
- Maintain a hardware catalog of C++ drivers/capabilities
- Create a developer's guide to explain the overall code structure, conventions, major
data classes, and guides for adding new hardware and implementations
- Provide an API reference for the most important classes for developers. Specifically,
these should be classes like SettingsStorage, HardwareObject, etc that are used
throughout the code. These classes should have Doxygen-style annotations in headers for
autogeneration with breathe.

### [Packaging and Binary Generation (Github Actions)](packaging-and-ci.md)

CMake-side work is complete: `cmake/Packaging.cmake` produces release-only DEB,
RPM, DMG, NSIS, and TGZ/ZIP packages via CPack; `cmake/QtDeployment.cmake` wires
`windeployqt`/`macdeployqt` as install hooks; and `.github/workflows/release.yml`
defines five jobs (linux-deb, linux-rpm, linux-appimage, macos-dmg, windows-nsis)
triggered by `workflow_dispatch` and `release: published`. **Remaining work is
testing and verification**: dispatch the workflow per-platform, iterate on
first-run issues (likely candidates: linuxdeploy library paths, macdeployqt's
Qwt resolution, windeployqt's runtime DLLs), and confirm the resulting packages
launch on clean VMs. See `packaging-and-ci.md` for the full strategy reference,
file map, and acceptance criteria.
