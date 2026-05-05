# Development Roadmap

Projects sorted by estimated complexity (smallest first). All are largely independent.

## Small

- **Latent `subKey` references in hardware-related code paths.** Two
  surviving uses of the literal `"subKey"` look like leftovers from the
  hardware-loadout naming scheme that the `cmakemigration` branch
  superseded. Each likely needs to be reconciled with the current
  `(hwType.label → driver)` model:
  - `src/gui/lif/gui/liflasercontroldoublespinbox.cpp:16` reads
    `s.value(QString("subKey"), …)` from QSettings inside the `lifLaser`
    group. The intent appears to be picking the active LIF-laser
    *driver* — but the runtime hardware system no longer stores driver
    selections under a `subKey` group. Either rewrite this against the
    current hardware selection plumbing or delete the dialog.
  - `tests/tst_settingsstoragetest.cpp:299, 367` use `"subKey"` as an
    arbitrary settings group name. These appear to be stale fixture
    names that predate the loadout system; verify they still test
    something meaningful and rename or remove as appropriate.
  Both surfaced during the bundle 01 audit of the `hardware.csv`
  `subKey → driver` rename and were intentionally left out of that
  bundle's scope.

## Medium

- [Python module refresh](python-module-refresh/00-overview.md) — bring the
  companion `blackchirp` Python package up to date with the v2 schema, add LIF
  read/processing support and an example notebook, and refresh the rendered
  documentation. Includes a prerequisite C++ enum-string migration and reader-
  hardening pass.

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
