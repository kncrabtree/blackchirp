# Development Roadmap

Projects sorted by estimated complexity (smallest first). All are largely independent.

## Medium

### Sirah Cobra integration refresh

A new Sirah Cobra dye laser is coming online in late May / early June
2026. Use that hardware as the trigger for revisiting the
`SirahCobra` driver: the existing TODO in
`src/hardware/core/liflaser/sirahcobra.cpp:112` flags that the
external-stage communication settings need a different solution
(separate baud / read terminator from the laser comm port). Plan the
scope of the refresh — driver-side, settings-storage, and
hardware-config UI — once the new instrument is on the bench. No
work in this directory until that planning happens.

## Large

None.

## Pre-Release

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
