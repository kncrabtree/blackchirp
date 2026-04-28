# Development Roadmap

Projects sorted by estimated complexity (smallest first). All are largely independent.

## Small

None

## Medium

None.

## Large

None.

## Pre-Release

### [Documentation Revision](documentation-revision.md)

The sphinx/breathe documentation is outdated and needs to be updated for the
`cmakemigration` branch. The goals are:

- Improve the readme and program summary for the landing page and Github
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
