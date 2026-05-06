# Development Roadmap

Projects sorted by estimated complexity (smallest first). All are largely independent.

## Medium

- [Python module refresh](python-module-refresh/00-overview.md) — bring the
  companion `blackchirp` Python package up to date with the v2 schema, add LIF
  read/processing support and an example notebook, and refresh the rendered
  documentation. Includes a prerequisite C++ enum-string migration and reader-
  hardening pass.

- **Multi-experiment FID coaveraging in the Python module.** A
  commented-out `BCFid.create_coaverage` classmethod sketch lives in
  `python/blackchirp/src/blackchirp/bcfid.py`. The intended feature
  is a coaverage of the FID from multiple experiments (with optional
  per-FID time-domain alignment via cross-correlation against a
  reference window). The sketch needs a proper design pass before
  it can ship; among the open considerations:
  - Should the entry point live on `BCFid` (classmethod) or on
    `BCFTMW` (factory that returns a `BCFid`-shaped object)?
  - How is compatibility between the input FIDs validated
    (matching `spacing`, `size`, `probefreq`, `sideband`, vmult
    treatment, frame count)? What is the policy when one or more
    of these disagree?
  - Phase-correction window: the existing sketch uses raw integer
    offsets; a user-facing API likely wants μs values resolved
    against `spacing`. How is the reference FID chosen — first in
    the list, highest-shot-count, or user-specified?
  - Output `vmult`: the existing sketch reuses the first input's
    `vmult`, which is wrong if the inputs were taken on different
    digitizers or scope settings. The output must rescale or
    refuse.
  - Should the API accept already-loaded `BCFid` / `BCExperiment`
    objects, paths, or both? The current sketch takes parallel
    `explist`/`pathlist` arguments that are not self-consistent.
  - Does coaveraging need to support multi-frame FIDs (current
    sketch flattens silently)?
  - Where do the C++-side semantics live — does Blackchirp itself
    have an analogous "coaverage across experiments" concept that
    this should mirror, or is the Python module the canonical
    home?

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
